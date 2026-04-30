import os
import gc
import re
import json
import time
import wave
import queue
import threading
import subprocess
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor


import numpy as np
import sounddevice as sd
import Jetson.GPIO as GPIO


from faster_whisper import WhisperModel
from openai import OpenAI
from gtts import gTTS


# Try to import the teammate's voice emotion module
try:
    from emotion_engine import EmotionEngine
except ImportError:
    EmotionEngine = None
    print("⚠️ Warning: emotion_engine.py not found. Voice emotion disabled.", flush=True)


warnings.filterwarnings("ignore")


# =========================================================
# Configuration
# =========================================================


# --- Cloud API Configuration ---
API_KEY = "sk-c1ccce3719bb4dca9615613f743c17ef"
BASE_URL = "https://api.deepseek.com"
CLOUD_MODEL = "deepseek-chat"


# --- Hardware / Path Configuration ---
BUTTON_PIN = 40          # BOARD numbering
INPUT_DEVICE = 0         # USB PnP Audio Device: Audio (hw:0,0)
CHANNELS = 1
MAX_RECORD_SECONDS = 10


WHISPER_SIZE = "base"
FUSION_JSON_PATH = os.path.expanduser("~/libreface_test/latest_fused.json")
UI_STATE_PATH = os.path.expanduser("~/emoq_ui/ui_state.json")


# --- Button debounce settings ---
BUTTON_PRESS_DEBOUNCE_SEC = 0.08
BUTTON_RELEASE_DEBOUNCE_SEC = 0.12
BUTTON_STARTUP_RELEASE_SEC = 0.20
# After conversation, require button to be released this long before re-arming
POST_CONV_RELEASE_GUARD_SEC = 0.30


# --- State Machine / UX Settings ---
TEST_MODE = True


if TEST_MODE:
    LOW_MOOD_WINDOW_SEC = 10
    LOW_MOOD_RATIO = 0.6
    PROMPT_COOLDOWN_SEC = 20
    PROMPT_WAIT_TIMEOUT_SEC = 10
    MONITOR_POLL_SEC = 0.1
else:
    LOW_MOOD_WINDOW_SEC = 120
    LOW_MOOD_RATIO = 0.75
    PROMPT_COOLDOWN_SEC = 3 * 60 * 60
    PROMPT_WAIT_TIMEOUT_SEC = 20
    MONITOR_POLL_SEC = 0.1


NEGATIVE_EMOTIONS = {"sad", "fear", "angry", "disgust"}


# --- Streaming TTS settings ---
STREAM_MIN_CHUNK_CHARS = 12
SENTENCE_TERMINATORS = {".", "!", "?", "。", "！", "？", "\n"}


# =========================================================
# Client init
# =========================================================


try:
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    print("☁️ Cloud Client (DeepSeek) connected successfully.", flush=True)
except Exception as e:
    print(f"❌ Cloud Client configuration failed: {e}", flush=True)
    client = None


# =========================================================
# UI helper
# =========================================================


def write_ui(mode="idle", robot_face="idle", user_text="", bot_text="",
             prompt_text="", who="Unknown User", user_emotion="neutral",
             status=""):
    try:
        os.makedirs(os.path.dirname(UI_STATE_PATH), exist_ok=True)
        payload = {
            "mode": mode,
            "robot_face": robot_face,
            "user_text": user_text,
            "bot_text": bot_text,
            "prompt_text": prompt_text,
            "who": who,
            "user_emotion": user_emotion,
            "status": status,
            "ts": time.time(),
        }
        tmp = UI_STATE_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, UI_STATE_PATH)
    except Exception as e:
        print(f"⚠️ UI write error: {e}", flush=True)




def update_idle_ui():
    vis_context = get_visual_context()
    robot_face = "smile" if vis_context["emotion"] == "sad" and vis_context["who"] != "Unknown User" else "idle"
    write_ui(
        mode="idle",
        robot_face=robot_face,
        user_text="",
        bot_text="",
        prompt_text="",
        who=vis_context["who"],
        user_emotion=vis_context["emotion"],
        status="monitoring"
    )


# =========================================================
# Helper functions
# =========================================================


def get_visual_context():
    if not os.path.exists(FUSION_JSON_PATH):
        return {"who": "Unknown User", "emotion": "neutral", "conf": 0.0}


    try:
        with open(FUSION_JSON_PATH, "r") as f:
            data = json.load(f)


        who = data.get("who", "Unknown")
        face_emotion = data.get("emotion", "neutral")
        overall_conf = data.get("overall_conf", 0.0)


        if overall_conf < 0.3 or who == "Unknown":
            who = "Unknown User"


        return {
            "who": who,
            "emotion": face_emotion.lower() if isinstance(face_emotion, str) else "neutral",
            "conf": overall_conf
        }
    except Exception:
        return {"who": "Unknown User", "emotion": "neutral", "conf": 0.0}




def speak(text):
    """Single-shot blocking TTS for prompt_state."""
    if not text or len(text.strip()) == 0:
        return


    try:
        print("🔊 EmoQ Speaking...", flush=True)
        tts = gTTS(text=text, lang='en', slow=False)
        audio_file = "response.mp3"
        tts.save(audio_file)
        os.system(f"mpg123 -a plughw:1,0 -q {audio_file}")
        if os.path.exists(audio_file):
            os.remove(audio_file)
    except Exception as e:
        print(f"⚠️ TTS Audio Playback Error: {e}", flush=True)




# =========================================================
# Streaming TTS pipeline
# =========================================================


def split_into_sentences(buffer):
    sentences = []
    current = ""
    last_cut = 0


    for i, ch in enumerate(buffer):
        current += ch
        if ch in SENTENCE_TERMINATORS:
            stripped = current.strip()
            if stripped:
                sentences.append(stripped)
            current = ""
            last_cut = i + 1


    remaining = buffer[last_cut:]
    return sentences, remaining




def _synth_and_play(sentence, idx, timing_log):
    if not sentence or not sentence.strip():
        return


    audio_file = f"response_chunk_{idx}.mp3"
    try:
        t_synth_start = time.time()
        tts = gTTS(text=sentence, lang='en', slow=False)
        tts.save(audio_file)
        t_synth_end = time.time()

        synth_dt = t_synth_end - t_synth_start

        if idx == 0:
            timing_log['first_tts_synth_dt'] = synth_dt
            timing_log['first_audio_play_start'] = time.time()
            print(f"⏱️ [TIMING] First TTS chunk synth (gTTS): {synth_dt:.3f}s", flush=True)
        else:
            print(f"⏱️ [TIMING] Chunk {idx} synth: {synth_dt:.3f}s", flush=True)

        subprocess.run(
            ["mpg123", "-a", "plughw:1,0", "-q", audio_file],
            check=False
        )
    except Exception as e:
        print(f"⚠️ Stream TTS chunk error: {e}", flush=True)
    finally:
        if os.path.exists(audio_file):
            try:
                os.remove(audio_file)
            except Exception:
                pass




def stream_speak_worker(sentence_q, done_event, timing_log):
    idx = 0
    while True:
        try:
            item = sentence_q.get(timeout=0.5)
        except queue.Empty:
            if done_event.is_set() and sentence_q.empty():
                break
            continue


        if item is None:
            break


        _synth_and_play(item, idx, timing_log)
        idx += 1
        sentence_q.task_done()




def ask_deepseek(text, final_emotion, identity, system_prompt):
    if client is None:
        print("⚠️ Cloud client is not available.", flush=True)
        return None


    try:
        final_message = f"[User Identity: {identity}] [User Emotion: {final_emotion}] User said: {text}"
        response = client.chat.completions.create(
            model=CLOUD_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": final_message},
            ],
            stream=True,
            temperature=1.0,
            max_tokens=200
        )
        return response
    except Exception as e:
        print(f"⚠️ Network Error: {e}", flush=True)
        return None




# =========================================================
# Button helpers (EDGE-DETECTION VERSION)
# =========================================================


def button_pressed():
    """Button is active-low (with PUD_UP enabled)."""
    return GPIO.input(BUTTON_PIN) == 0




def wait_until_button_released(min_release_sec=BUTTON_STARTUP_RELEASE_SEC,
                                timeout_sec=5.0):
    """
    Block until button has been STABLY released for at least min_release_sec.
    Used at startup AND after every conversation to prevent ghost triggers.

    Defensive timeout: if for some reason the button reads "pressed" forever
    (hardware glitch, broken wire, etc), don't hang the whole state machine.
    Logs a warning and proceeds.
    """
    start = None
    t_begin = time.time()
    while True:
        if not button_pressed():
            if start is None:
                start = time.time()
            elif time.time() - start >= min_release_sec:
                return
        else:
            start = None

        # Defensive timeout
        if time.time() - t_begin > timeout_sec:
            print(f"⚠️ wait_until_button_released TIMEOUT after {timeout_sec}s "
                  f"(GPIO stuck reading pressed). Proceeding anyway.", flush=True)
            return

        time.sleep(0.01)




def wait_for_press_edge(monitoring_callback=None, poll_interval=0.02):
    """
    Wait for a real RELEASED -> PRESSED transition.

    Critical fix: this function REQUIRES the button to first be observed as released
    before counting any press as valid. Eliminates the ghost-trigger bug.

    Returns:
        "PRESSED" if a real press edge was detected,
        "ABORT"   if monitoring_callback returned True.
    """
    # Phase 1: confirm button is currently released (with debounce)
    release_start = None
    while True:
        if monitoring_callback is not None:
            if monitoring_callback():
                return "ABORT"

        if not button_pressed():
            if release_start is None:
                release_start = time.time()
            elif time.time() - release_start >= BUTTON_RELEASE_DEBOUNCE_SEC:
                break
        else:
            release_start = None
        time.sleep(poll_interval)

    # Phase 2: now wait for an actual press (with debounce)
    press_start = None
    while True:
        if monitoring_callback is not None:
            if monitoring_callback():
                return "ABORT"

        if button_pressed():
            if press_start is None:
                press_start = time.time()
            elif time.time() - press_start >= BUTTON_PRESS_DEBOUNCE_SEC:
                return "PRESSED"
        else:
            press_start = None
        time.sleep(poll_interval)




def button_released_stably():
    """Return True only if button stays released for debounce window."""
    start = None
    while True:
        if not button_pressed():
            if start is None:
                start = time.time()
            elif time.time() - start >= BUTTON_RELEASE_DEBOUNCE_SEC:
                return True
        else:
            return False
        time.sleep(0.005)




# =========================================================
# Audio helpers
# =========================================================


def get_input_samplerate():
    try:
        device_info = sd.query_devices(INPUT_DEVICE, 'input')
        samplerate = int(device_info['default_samplerate'])
        print(f"🎤 Using input device index: {INPUT_DEVICE}", flush=True)
        print(f"🎤 Input device name: {device_info['name']}", flush=True)
        print(f"🎤 Device default samplerate: {samplerate}", flush=True)
        return samplerate
    except Exception as e:
        print(f"⚠️ Could not query input device samplerate: {e}", flush=True)
        return 44100




def record_while_button_held(output_file="temp.wav"):
    samplerate = get_input_samplerate()
    print("🎙️ Recording... hold button while speaking.", flush=True)


    frames = []
    start_time = time.time()


    def callback(indata, frames_count, time_info, status):
        if status:
            print(f"⚠️ Audio callback status: {status}", flush=True)
        frames.append(indata.copy())


    with sd.InputStream(
        samplerate=samplerate,
        channels=CHANNELS,
        dtype='int16',
        callback=callback,
        device=INPUT_DEVICE
    ):
        while True:
            if time.time() - start_time > MAX_RECORD_SECONDS:
                print("⏱️ Max recording time reached.", flush=True)
                break
            if button_released_stably():
                print("🛑 Stable button release detected. Stopping recording.", flush=True)
                break
            time.sleep(0.01)


    if not frames:
        return None


    audio_data = np.concatenate(frames, axis=0)
    duration = len(audio_data) / samplerate
    print(f"🕒 Recorded {duration:.2f} seconds", flush=True)


    if duration < 0.8:
        print("⚠️ Recording too short.", flush=True)
        return None


    with wave.open(output_file, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio_data.tobytes())


    print(f"💾 Saved recording to {output_file}", flush=True)
    return output_file




# =========================================================
# Mood / prompt helpers
# =========================================================


def play_prompt():
    prompt_text = "You seem a bit down. If you want to talk, press and hold the button."
    print(f"💬 Prompt: {prompt_text}", flush=True)
    speak(prompt_text)




def is_negative_emotion(emotion):
    return emotion in NEGATIVE_EMOTIONS




def emotion_sample_valid(vis_context):
    who = vis_context["who"]
    conf = vis_context["conf"]
    return who != "Unknown User" and conf >= 0.5




def should_trigger_prompt(emotion_history, last_prompt_time):
    now = time.time()
    if now - last_prompt_time < PROMPT_COOLDOWN_SEC:
        return False
    if not emotion_history:
        return False
    total = len(emotion_history)
    negative_count = sum(1 for _, emo in emotion_history if is_negative_emotion(emo))
    ratio = negative_count / total if total > 0 else 0.0
    print(f"📉 Mood trend check: {negative_count}/{total} negative = {ratio:.2f}", flush=True)
    return ratio >= LOW_MOOD_RATIO




# =========================================================
# Parallel inference helpers
# =========================================================


def run_whisper_transcribe(audio_model, audio_path):
    """Run Whisper in a thread. Returns transcribed text."""
    try:
        segments, info = audio_model.transcribe(audio_path, beam_size=5)
        text = " ".join([segment.text for segment in segments]).strip()
        return text
    except Exception as e:
        print(f"⚠️ Whisper error: {e}", flush=True)
        return ""




def run_voice_emotion(emo_engine, audio_path):
    """Run voice emotion engine in a thread. Returns emotion string."""
    if emo_engine is None:
        return "neutral"
    try:
        voice_emotion = emo_engine.predict(audio_path)
        if isinstance(voice_emotion, str):
            return voice_emotion.lower()
        return "neutral"
    except Exception as e:
        print(f"⚠️ Voice emotion error: {e}", flush=True)
        return "neutral"




# =========================================================
# State handlers
# =========================================================


def monitoring_state(emotion_history, last_prompt_time):
    """
    Passive monitoring with EDGE-DETECTED button trigger.
    Button must first be observed as released BEFORE any press counts as valid.
    """
    print("🟢 State = MONITORING", flush=True)
    print("👀 Monitoring face/emotion trend...", flush=True)
    print("🖲️ User may also press button anytime to start talking.", flush=True)

    monitor_state = {
        "emotion_history": emotion_history,
        "last_prompt_time": last_prompt_time,
        "trigger_prompt": False,
    }

    def monitor_tick():
        vis_context = get_visual_context()
        robot_face = "smile" if vis_context["emotion"] == "sad" and vis_context["who"] != "Unknown User" else "idle"
        write_ui(
            mode="idle",
            robot_face=robot_face,
            user_text="",
            bot_text="",
            prompt_text="",
            who=vis_context["who"],
            user_emotion=vis_context["emotion"],
            status="monitoring"
        )

        now = time.time()
        eh = monitor_state["emotion_history"]
        while eh and (now - eh[0][0] > LOW_MOOD_WINDOW_SEC):
            eh.popleft()

        if emotion_sample_valid(vis_context):
            eh.append((now, vis_context["emotion"]))

        if should_trigger_prompt(eh, monitor_state["last_prompt_time"]):
            print("💡 Sustained low mood detected -> entering PROMPT", flush=True)
            monitor_state["trigger_prompt"] = True
            return True

        return False

    result = wait_for_press_edge(monitoring_callback=monitor_tick,
                                 poll_interval=MONITOR_POLL_SEC)

    if result == "PRESSED":
        print("🖲️ Real button-press edge detected -> entering CONVERSATION", flush=True)
        return "CONVERSATION"
    else:
        return "PROMPT"




def prompt_state():
    print("🟡 State = PROMPT", flush=True)
    vis_context = get_visual_context()
    prompt_text = "You seem a bit down. If you want to talk, press and hold the button."

    write_ui(
        mode="prompt",
        robot_face="smile",
        user_text="",
        bot_text="",
        prompt_text=prompt_text,
        who=vis_context["who"],
        user_emotion=vis_context["emotion"],
        status="waiting for button"
    )

    play_prompt()

    start_wait = time.time()
    print("⌛ Waiting for user to press button (edge-detected)...", flush=True)

    def prompt_tick():
        if time.time() - start_wait > PROMPT_WAIT_TIMEOUT_SEC:
            return True
        return False

    result = wait_for_press_edge(monitoring_callback=prompt_tick, poll_interval=0.05)

    if result == "PRESSED":
        print("🖲️ Real button-press edge during PROMPT -> entering CONVERSATION", flush=True)
        return "CONVERSATION"
    else:
        print("⌛ Prompt timed out. Returning to monitoring.", flush=True)
        update_idle_ui()
        return "MONITORING"




def conversation_state(audio_model, emo_engine, system_prompt):
    """
    Button-driven voice interaction with:
    - PARALLEL Whisper + Voice emotion (saves ~3s)
    - Streaming TTS (early playback)
    - Detailed timing instrumentation
    - Post-conversation release guard (prevents ghost-trigger)
    """
    print("🔵 State = CONVERSATION", flush=True)


    vis_context = get_visual_context()
    write_ui(
        mode="conversation",
        robot_face="listening",
        user_text="Listening...",
        bot_text="",
        prompt_text="",
        who=vis_context["who"],
        user_emotion=vis_context["emotion"],
        status="recording"
    )


    audio_path = record_while_button_held("temp.wav")
    if not audio_path:
        print("⚠️ No audio captured.", flush=True)
        update_idle_ui()
        wait_until_button_released(POST_CONV_RELEASE_GUARD_SEC)
        return "MONITORING"


    # ===================== TIMING START =====================
    T0 = time.time()
    print("=" * 60, flush=True)
    print("⏱️ [TIMING] T0 = button released (recording done)", flush=True)
    print("=" * 60, flush=True)


    # --- Stage 1+2: PARALLEL Whisper + Voice emotion ---
    t_parallel_start = time.time()
    print("⚡ [PARALLEL] Launching Whisper + Voice emotion concurrently...", flush=True)

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_whisper = executor.submit(run_whisper_transcribe, audio_model, audio_path)
        future_voice_emo = executor.submit(run_voice_emotion, emo_engine, audio_path)

        text = future_whisper.result()
        t_whisper_end = time.time()
        whisper_dt = t_whisper_end - t_parallel_start
        print(f"⏱️ [TIMING] Whisper done: {whisper_dt:.3f}s (parallel)", flush=True)

        voice_emotion = future_voice_emo.result()
        t_voice_emo_end = time.time()
        voice_emo_dt = t_voice_emo_end - t_parallel_start
        print(f"⏱️ [TIMING] Voice emotion done: {voice_emo_dt:.3f}s (parallel)", flush=True)

    t_parallel_end = time.time()
    parallel_total = t_parallel_end - t_parallel_start
    print(f"⏱️ [TIMING] Parallel block total: {parallel_total:.3f}s "
          f"(saved ~{(whisper_dt + voice_emo_dt) - parallel_total:.3f}s vs serial)", flush=True)
    print(f"📝 Transcription: {repr(text)}", flush=True)


    if not text or len(text) < 2:
        print("⚠️ No valid speech detected.", flush=True)
        update_idle_ui()
        wait_until_button_released(POST_CONV_RELEASE_GUARD_SEC)
        return "MONITORING"


    # --- Stage 3: Visual context (file read, ~0s) ---
    t_vis_start = time.time()
    vis_context = get_visual_context()
    face_emotion = vis_context["emotion"]
    identity = vis_context["who"]
    face_conf = vis_context["conf"]
    t_vis_end = time.time()
    print(f"⏱️ [TIMING] Visual context read: {t_vis_end - t_vis_start:.3f}s", flush=True)


    # Fusion logic (unchanged)
    if voice_emotion in {"happy", "angry"}:
        final_emotion = voice_emotion
        fusion_source = "Voice-Priority"
    elif voice_emotion == "sad":
        if face_conf > 0.6 and face_emotion in {"sad", "angry"}:
            final_emotion = "sad"
            fusion_source = "Voice-Confirmed-By-Face"
        else:
            final_emotion = "neutral"
            fusion_source = "Sad-Downgraded-To-Neutral"
    elif face_conf > 0.6:
        final_emotion = face_emotion
        fusion_source = "Face-Fallback"
    else:
        final_emotion = "neutral"
        fusion_source = "Neutral-Fallback"


    print(f"📊 Fusion: Face={face_emotion}({face_conf:.2f}) Voice={voice_emotion} -> {final_emotion} ({fusion_source})", flush=True)


    write_ui(
        mode="conversation",
        robot_face="thinking",
        user_text=text,
        bot_text="",
        prompt_text="",
        who=identity,
        user_emotion=final_emotion,
        status="thinking"
    )


    # --- Stage 4: DeepSeek API call ---
    t_llm_request_start = time.time()
    stream = ask_deepseek(text, final_emotion, identity, system_prompt)
    t_llm_request_end = time.time()
    llm_request_dt = t_llm_request_end - t_llm_request_start
    print(f"⏱️ [TIMING] DeepSeek API connect: {llm_request_dt:.3f}s", flush=True)


    if stream:
        timing_log = {}
        sentence_q = queue.Queue()
        done_event = threading.Event()
        player_thread = threading.Thread(
            target=stream_speak_worker,
            args=(sentence_q, done_event, timing_log),
            daemon=True
        )
        player_thread.start()


        print("🤖 EmoQ: ", end="", flush=True)
        full_response = ""
        text_buffer = ""
        first_token_time = None
        first_sentence_time = None


        try:
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if not delta:
                    continue


                if first_token_time is None:
                    first_token_time = time.time()
                    print(f"\n⏱️ [TIMING] LLM first token: {first_token_time - t_llm_request_end:.3f}s after request", flush=True)
                    print("🤖 EmoQ: ", end="", flush=True)


                print(delta, end="", flush=True)
                full_response += delta
                text_buffer += delta


                sentences, text_buffer = split_into_sentences(text_buffer)
                for sent in sentences:
                    if first_sentence_time is None:
                        first_sentence_time = time.time()
                        print(f"\n⏱️ [TIMING] First complete sentence ready: {first_sentence_time - first_token_time:.3f}s after first token", flush=True)
                        print(f"⏱️ [TIMING] First sentence: {repr(sent)}", flush=True)
                        print("🤖 EmoQ: ", end="", flush=True)
                    sentence_q.put(sent)


                write_ui(
                    mode="conversation",
                    robot_face="thinking",
                    user_text=text,
                    bot_text=full_response,
                    prompt_text="",
                    who=identity,
                    user_emotion=final_emotion,
                    status="responding"
                )


            print("", flush=True)
            tail = text_buffer.strip()
            if tail:
                sentence_q.put(tail)


        except Exception as e:
            print(f"\n⚠️ Stream read error: {e}", flush=True)


        write_ui(
            mode="conversation",
            robot_face="smile",
            user_text=text,
            bot_text=full_response,
            prompt_text="",
            who=identity,
            user_emotion=final_emotion,
            status="speaking"
        )


        sentence_q.put(None)
        done_event.set()
        player_thread.join(timeout=60)


        # ===================== TIMING SUMMARY =====================
        T_END = time.time()
        first_audio_play = timing_log.get('first_audio_play_start', T_END)
        print("=" * 60, flush=True)
        print("⏱️ [TIMING SUMMARY]", flush=True)
        print(f"   T0 (button release)       -> Parallel block done: {t_parallel_end - T0:.3f}s", flush=True)
        print(f"     [Whisper alone:         {whisper_dt:.3f}s]", flush=True)
        print(f"     [Voice emo alone:       {voice_emo_dt:.3f}s]", flush=True)
        print(f"   Parallel done             -> LLM request sent:    {t_llm_request_end - t_parallel_end:.3f}s", flush=True)
        if first_token_time:
            print(f"   LLM request               -> first token:         {first_token_time - t_llm_request_end:.3f}s", flush=True)
        if first_sentence_time:
            print(f"   First token               -> first sentence:      {first_sentence_time - first_token_time:.3f}s", flush=True)
            print(f"   First sentence ready      -> first TTS done:      {timing_log.get('first_tts_synth_dt', 0):.3f}s", flush=True)
        print(f"   ----- TOTAL T0 -> first audio playback start: {first_audio_play - T0:.3f}s -----", flush=True)
        print("=" * 60, flush=True)


    else:
        print("🤖 (Network Error)", flush=True)
        write_ui(
            mode="conversation",
            robot_face="idle",
            user_text=text,
            bot_text="Network error.",
            prompt_text="",
            who=identity,
            user_emotion=final_emotion,
            status="network error"
        )


    gc.collect()
    update_idle_ui()

    # CRITICAL: Guarantee button is released before returning to MONITORING
    print(f"🔒 Post-conversation guard: ensuring button is released for {POST_CONV_RELEASE_GUARD_SEC}s...", flush=True)
    wait_until_button_released(POST_CONV_RELEASE_GUARD_SEC)
    print("✅ Button confirmed released. Returning to MONITORING.", flush=True)
    return "MONITORING"




# =========================================================
# Main
# =========================================================


def main():
    print("🚀 Starting EmoQ (FIXED + PARALLELIZED + TIMED)...", flush=True)
    print(f"👂 Loading Local Hearing (Faster-Whisper {WHISPER_SIZE})...", flush=True)


    print("⚙️ Current mode settings:", flush=True)
    print(f"   TEST_MODE = {TEST_MODE}", flush=True)
    print(f"   LOW_MOOD_WINDOW_SEC = {LOW_MOOD_WINDOW_SEC}", flush=True)
    print(f"   LOW_MOOD_RATIO = {LOW_MOOD_RATIO}", flush=True)
    print(f"   PROMPT_COOLDOWN_SEC = {PROMPT_COOLDOWN_SEC}", flush=True)
    print(f"   PROMPT_WAIT_TIMEOUT_SEC = {PROMPT_WAIT_TIMEOUT_SEC}", flush=True)
    print(f"   MONITOR_POLL_SEC = {MONITOR_POLL_SEC}", flush=True)


    audio_model = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")


    emo_engine = None
    if EmotionEngine:
        emo_engine = EmotionEngine()


    # === GPIO setup ===
    # Jetson.GPIO ignores pull_up_down parameter; the button hardware already has
    # external pull-up (verified by diagnostic: floating-mode reads stable 1 when
    # released, 0 when pressed). So no internal pull configuration is needed.
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(BUTTON_PIN, GPIO.IN)
    print("🔌 GPIO configured (external pull-up assumed; active-LOW button)", flush=True)


    print("🧷 Waiting for button to be released before starting...", flush=True)
    wait_until_button_released()


    system_prompt = (
        "You are EmoQ, an empathetic companion robot. "
        "You will receive the user's name and their current emotional state. "
        "If the user is not 'Unknown User', address them by their name occasionally. "
        "Respond warmly, empathetically, and concisely in English (1-2 sentences). "
        "Adjust your tone based on their emotion."
    )


    print("=" * 55, flush=True)
    print("🤖 EmoQ Online (Monitoring / Prompt / Conversation)", flush=True)
    print("=" * 55, flush=True)


    emotion_history = deque()
    last_prompt_time = 0
    state = "MONITORING"


    update_idle_ui()


    try:
        while True:
            try:
                if state == "MONITORING":
                    state = monitoring_state(emotion_history, last_prompt_time)


                elif state == "PROMPT":
                    last_prompt_time = time.time()
                    state = prompt_state()


                elif state == "CONVERSATION":
                    state = conversation_state(audio_model, emo_engine, system_prompt)


                else:
                    print(f"⚠️ Unknown state: {state}. Resetting to MONITORING.", flush=True)
                    state = "MONITORING"


            except Exception as e:
                print(f"⚠️ Main loop error: {e}", flush=True)
                update_idle_ui()
                state = "MONITORING"
                time.sleep(1)


    except KeyboardInterrupt:
        print("\n👋 Exiting...", flush=True)


    finally:
        GPIO.cleanup()




if __name__ == "__main__":
    main()