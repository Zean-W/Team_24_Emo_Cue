import speech_recognition as sr
# ❌ No longer using import whisper
# ✅ Using faster_whisper for performance
from faster_whisper import WhisperModel
import ollama
import warnings
import os
import time
import gc

# Import teammate's emotion recognition module
from emotion_engine import EmotionEngine

# Ignore warnings
warnings.filterwarnings("ignore")

# --- Configuration Area ---
MIC_INDEX = 0             # Check your mic index with ll_mic.py if needed
LLM_MODEL = "qwen2:1.5b"  
# Faster-Whisper model size
WHISPER_SIZE = "tiny"     

def main():
    print("\n🚀 Starting EmoQ System (Faster-Whisper Edition)...")
    
    # --- 1. Load Faster-Whisper ---
    # Critical Optimization: compute_type="int8"
    # This compresses the model to minimize RAM usage and CPU load.
    print(f"👂 Loading Faster-Whisper ({WHISPER_SIZE} int8)...")
    try:
        # device="cpu" is recommended for stability on Jetson shared memory
        # You can try device="cuda" if you have plenty of RAM, but cpu is safer here.
        audio_model = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")
        print("✅ Faster-Whisper Ready")
    except Exception as e:
        print(f"❌ Failed to load Whisper: {e}")
        return

    # --- 2. Load Emotion Engine ---
    print("🎭 Loading Emotion Engine...")
    try:
        emo_engine = EmotionEngine()
        print("✅ Emotion Engine Ready")
    except Exception as e:
        print(f"⚠️ Failed to load Emotion Engine: {e}")
        emo_engine = None

    # --- 3. Initialize Microphone ---
    r = sr.Recognizer()
    r.energy_threshold = 1000
    r.dynamic_energy_threshold = True

    try:
        source = sr.Microphone(device_index=MIC_INDEX)
        print(f"✅ Microphone Ready (Device {MIC_INDEX})")
    except Exception as e:
        print(f"❌ Microphone Connection Failed: {e}")
        return

    system_prompt = (
        "You are EmoQ, an emotional companion robot for seniors. "
        "You will receive the user's input text along with their detected voice emotion. "
        "Adapt your tone and response to match or comfort their emotion. "
        "Keep responses warm, concise, and helpful."
    )

    print("\n" + "="*50)
    print(f"🤖 EmoQ System Online (Performance Mode)")
    print("🗣️  Please speak...")
    print("="*50)

    # --- 4. Main Loop ---
    with source:
        r.adjust_for_ambient_noise(source, duration=1)
        
        while True:
            try:
                print("\n👂 Listening...", end="\r")
                audio = r.listen(source, timeout=None, phrase_time_limit=6)
                print("🧠 Processing...       ", end="\r")

                # Save audio to temp file
                with open("temp.wav", "wb") as f:
                    f.write(audio.get_wav_data())

                # Step 1: Faster-Whisper Transcription
                # faster-whisper returns segments (generator), need to join them
                segments, info = audio_model.transcribe("temp.wav", beam_size=5)
                text = " ".join([segment.text for segment in segments]).strip()

                if not text or len(text) < 2:
                    continue

                # Step 2: Emotion Recognition
                detected_emotion = "neutral"
                if emo_engine:
                    detected_emotion = emo_engine.predict("temp.wav")
                
                # Print Result in English for Demo
                print(f"👤 User: {text}")
                print(f"🎭 Emotion: [{detected_emotion}]")

                # Aggressive Garbage Collection (Prevent OOM)
                gc.collect()

                # Step 3: Send to Ollama
                final_user_message = f"[User Emotion: {detected_emotion}] User said: {text}"
                print("🤖 EmoQ: ", end="")
                
                stream = ollama.chat(
                    model=LLM_MODEL,
                    messages=[
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': final_user_message}
                    ],
                    stream=True,
                    # CRITICAL: Limit context window to save RAM
                    options={'num_ctx': 256} 
                )

                for chunk in stream:
                    content = chunk['message']['content']
                    print(content, end='', flush=True)
                print("") 

            except KeyboardInterrupt:
                print("\n👋 Exiting program...")
                break
            except Exception as e:
                print(f"\n⚠️ Error: {e}")
                gc.collect() 
                continue

if __name__ == "__main__":
    main()