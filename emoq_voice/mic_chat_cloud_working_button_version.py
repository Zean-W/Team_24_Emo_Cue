import os
import gc
import json
import time
import wave
import warnings

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

# --- Cloud API Configuration ---
API_KEY = "sk-c1ccce3719bb4dca9615613f743c17ef"
BASE_URL = "https://api.deepseek.com"
CLOUD_MODEL = "deepseek-chat"

try:
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    print("☁️ Cloud Client (DeepSeek) connected successfully.", flush=True)
except Exception as e:
    print(f"❌ Cloud Client configuration failed: {e}", flush=True)
    client = None

# --- Hardware / Path Configuration ---
BUTTON_PIN = 40          # BOARD numbering
INPUT_DEVICE = 0         # USB PnP Audio Device: Audio (hw:0,0)
SAMPLE_RATE = 16000
CHANNELS = 1
MAX_RECORD_SECONDS = 10  # safety cap

WHISPER_SIZE = "base"
FUSION_JSON_PATH = os.path.expanduser("~/libreface_test/latest_fused.json")


def get_visual_context():
    """Read fused visual context JSON from teammate's pipeline."""
    if not os.path.exists(FUSION_JSON_PATH):
        return {"who": "Unknown", "emotion": "neutral", "conf": 0.0}

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
            "emotion": face_emotion,
            "conf": overall_conf
        }
    except Exception:
        return {"who": "Unknown", "emotion": "neutral", "conf": 0.0}


def speak(text):
    """Convert text to speech and play through speaker."""
    if not text or len(text.strip()) == 0:
        return

    try:
        print("🔊 EmoQ Speaking...", flush=True)
        tts = gTTS(text=text, lang='en', slow=False)
        audio_file = "response.mp3"
        tts.save(audio_file)

        # USB PnP Audio Device is hw:0,0
        os.system(f"mpg123 -a hw:0,0 -q {audio_file}")

        if os.path.exists(audio_file):
            os.remove(audio_file)
    except Exception as e:
        print(f"⚠️ TTS Audio Playback Error: {e}", flush=True)


def ask_deepseek(text, final_emotion, identity, system_prompt):
    """Send user input to DeepSeek cloud model."""
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
            temperature=1.3,
            max_tokens=300
        )
        return response
    except Exception as e:
        print(f"⚠️ Network Error: {e}", flush=True)
        return None


def button_pressed():
    """Button is active-low."""
    return GPIO.input(BUTTON_PIN) == 0


def wait_for_button_press():
    """Block until button is pressed."""
    print("🟢 Monitoring... Press and hold button to talk.", flush=True)
    while True:
        if button_pressed():
            time.sleep(0.05)  # debounce
            if button_pressed():
                return
        time.sleep(0.01)


def record_while_button_held(output_file="temp.wav", samplerate=SAMPLE_RATE):
    """
    Record audio while the button is held down.
    Stops when the button is released or max duration is reached.
    """
    print("🎤 Using input device index: 0 (USB PnP Audio Device)", flush=True)
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
        while button_pressed():
            if time.time() - start_time > MAX_RECORD_SECONDS:
                print("⏱️ Max recording time reached.", flush=True)
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
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(samplerate)
        wf.writeframes(audio_data.tobytes())

    print(f"💾 Saved recording to {output_file}", flush=True)
    return output_file


def main():
    print("🚀 Starting EmoQ (Button-to-Talk Version)...", flush=True)
    print(f"👂 Loading Local Hearing (Faster-Whisper {WHISPER_SIZE})...", flush=True)
    audio_model = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")

    emo_engine = None
    if EmotionEngine:
        emo_engine = EmotionEngine()

    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(BUTTON_PIN, GPIO.IN)

    system_prompt = (
        "You are EmoQ, an empathetic companion robot. "
        "You will receive the user's name and their current emotional state. "
        "If the user is not 'Unknown User', address them by their name occasionally. "
        "Respond warmly, empathetically, and concisely in English (1-2 sentences). "
        "Adjust your tone based on their emotion."
    )

    print("=" * 50, flush=True)
    print("🤖 EmoQ Online (Button-controlled Conversation)", flush=True)
    print("=" * 50, flush=True)

    try:
        while True:
            try:
                # Monitoring state
                wait_for_button_press()

                # Conversation state
                audio_path = record_while_button_held("temp.wav")
                if not audio_path:
                    print("⚠️ No audio captured.", flush=True)
                    continue

                print("🧠 Processing...", flush=True)

                segments, info = audio_model.transcribe(audio_path, beam_size=5)
                text = " ".join([segment.text for segment in segments]).strip()

                print(f"📝 Transcription raw: {repr(text)}", flush=True)

                if not text or len(text) < 2:
                    print("⚠️ No valid speech detected.", flush=True)
                    continue

                voice_emotion = "neutral"
                if emo_engine:
                    try:
                        voice_emotion = emo_engine.predict(audio_path)
                    except Exception as e:
                        print(f"⚠️ Voice emotion error: {e}", flush=True)
                        voice_emotion = "neutral"

                vis_context = get_visual_context()
                face_emotion = vis_context["emotion"]
                identity = vis_context["who"]
                face_conf = vis_context["conf"]

                # In conversation mode, prioritize voice emotion
                if voice_emotion and voice_emotion != "neutral":
                    final_emotion = voice_emotion
                    fusion_source = "Voice-Priority"
                elif face_conf > 0.6:
                    final_emotion = face_emotion
                    fusion_source = "Face-Fallback"
                else:
                    final_emotion = "neutral"
                    fusion_source = "Neutral-Fallback"

                print(f"👤 User ({identity}): {text}", flush=True)
                print(
                    f"📊 Fusion: [Face: {face_emotion}({face_conf:.2f}) | Voice: {voice_emotion}] "
                    f"-> Final: [{final_emotion}] ({fusion_source})",
                    flush=True
                )

                print("☁️ EmoQ Thinking...", flush=True)
                stream = ask_deepseek(text, final_emotion, identity, system_prompt)

                if stream:
                    print("🤖 EmoQ: ", end="", flush=True)
                    full_response = ""
                    for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            print(content, end="", flush=True)
                            full_response += content
                    print("", flush=True)
                    speak(full_response)
                else:
                    print("🤖 (Network Error)", flush=True)

                gc.collect()

            except Exception as e:
                print(f"⚠️ Error: {e}", flush=True)
                continue

    except KeyboardInterrupt:
        print("\n👋 Exiting...", flush=True)

    finally:
        GPIO.cleanup()


if __name__ == "__main__":
    main()
