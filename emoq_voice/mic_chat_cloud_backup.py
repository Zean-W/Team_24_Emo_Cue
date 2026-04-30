import speech_recognition as sr
from faster_whisper import WhisperModel
from openai import OpenAI
import warnings
import os
import gc
import json
import time
from gtts import gTTS

# Try to import the teammate's voice emotion module
try:
    from emotion_engine import EmotionEngine
except ImportError:
    EmotionEngine = None
    print("⚠️ Warning: emotion_engine.py not found. Voice emotion disabled.")

# Ignore warnings
warnings.filterwarnings("ignore")

# --- ☁️ Cloud API Configuration ---
API_KEY = "sk-c1ccce3719bb4dca9615613f743c17ef" 
BASE_URL = "https://api.deepseek.com"
CLOUD_MODEL = "deepseek-chat"

try:
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    print("☁️ Cloud Client (DeepSeek) connected successfully.")
except Exception as e:
    print(f"❌ Cloud Client configuration failed: {e}")

# --- Local Hardware & Path Configuration ---
# ⚠️ 注意：这里换了新声卡，请务必运行探测代码把 8 换成你的新 Index！
MIC_INDEX = 0           
WHISPER_SIZE = "tiny"     
# 队友输出的 JSON 文件路径
FUSION_JSON_PATH = os.path.expanduser("~/libreface_test/latest_fused.json")

def get_visual_context():
    """读取队友的视觉融合 JSON 文件"""
    if not os.path.exists(FUSION_JSON_PATH):
        return {"who": "Unknown", "emotion": "neutral", "conf": 0.0}
    
    try:
        with open(FUSION_JSON_PATH, "r") as f:
            data = json.load(f)
            
            who = data.get("who", "Unknown")
            face_emotion = data.get("emotion", "neutral")
            overall_conf = data.get("overall_conf", 0.0)
            
            # 根据队友建议：置信度太低，或者识别为 Unknown 时，做降级处理
            if overall_conf < 0.3 or who == "Unknown":
                who = "Unknown User"
                
            return {
                "who": who, 
                "emotion": face_emotion, 
                "conf": overall_conf
            }
    except Exception as e:
        # 防止正好在写入时读取导致的 JSON 解析报错
        return {"who": "Unknown", "emotion": "neutral", "conf": 0.0}

def speak(text):
    """将文本转换为语音并通过 USB 声卡播放"""
    if not text or len(text.strip()) == 0:
        return
    try:
        print("🔊 EmoQ Speaking...")
        # 英文发音，语速正常
        tts = gTTS(text=text, lang='en', slow=False)
        audio_file = "response.mp3"
        tts.save(audio_file)
        
        # 使用 mpg123 播放，-q 代表静默模式（不打印播放器乱码信息）
        os.system(f"mpg123 -a hw:0,0 -q {audio_file}")
        
        # 播放完毕后删掉临时音频文件
        if os.path.exists(audio_file):
            os.remove(audio_file)
    except Exception as e:
        print(f"⚠️ TTS Audio Playback Error: {e}")

def ask_deepseek(text, final_emotion, identity, system_prompt):
    """发送给 DeepSeek 云端大脑"""
    try:
        # 将身份和情绪都注入给大模型
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
        print(f"\n⚠️ Network Error: {e}")
        return None

def main():
    print("\n🚀 Starting EmoQ (Multimodal Fusion + TTS Version)...")
    
    print(f"👂 Loading Local Hearing (Faster-Whisper {WHISPER_SIZE})...")
    audio_model = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")
    
    emo_engine = None
    if EmotionEngine:
        emo_engine = EmotionEngine()

    r = sr.Recognizer()
    r.energy_threshold = 1000
    r.dynamic_energy_threshold = True
    
    try:
        source = sr.Microphone(device_index=MIC_INDEX)
    except Exception as e:
        print(f"\n❌ Error: 找不到 Index 为 {MIC_INDEX} 的麦克风。")
        print("请运行测试脚本找到 SSS1629 声卡的真实 Index，并修改代码中的 MIC_INDEX。")
        return

    # 升级后的 System Prompt
    system_prompt = (
        "You are EmoQ, an empathetic companion robot. "
        "You will receive the user's name and their current emotional state. "
        "If the user is not 'Unknown User', address them by their name occasionally. "
        "Respond warmly, empathetically, and concisely in English (1-2 sentences). "
        "Adjust your tone based on their emotion."
    )

    print("\n" + "="*50)
    print(f"🤖 EmoQ Online (DeepSeek V3 + Vision-Audio + TTS)")
    print("="*50)

    with source:
        r.adjust_for_ambient_noise(source, duration=1)
        while True:
            try:
                print("\n👂 Listening...", end="\r")
                audio = r.listen(source, timeout=None, phrase_time_limit=6)
                print("🧠 Processing...       ", end="\r")

                with open("temp.wav", "wb") as f:
                    f.write(audio.get_wav_data())

                # 1. 获取语音转文字
                segments, info = audio_model.transcribe("temp.wav", beam_size=5)
                text = " ".join([segment.text for segment in segments]).strip()

                if not text or len(text) < 2:
                    continue

                # 2. 获取声音情绪 (听觉)
                voice_emotion = "neutral"
                if emo_engine:
                    voice_emotion = emo_engine.predict("temp.wav")
                
                # 3. 获取视觉上下文 (视觉)
                vis_context = get_visual_context()
                face_emotion = vis_context["emotion"]
                identity = vis_context["who"]
                face_conf = vis_context["conf"]

                # 4. 🧠 多模态融合决策逻辑
                if face_conf > 0.6:
                    final_emotion = face_emotion
                    fusion_source = "Face-Led"
                else:
                    final_emotion = voice_emotion
                    fusion_source = "Voice-Led"

                print(f"👤 User ({identity}): {text}")
                print(f"📊 Fusion: [Face: {face_emotion}({face_conf:.2f}) | Voice: {voice_emotion}] -> Final: [{final_emotion}] ({fusion_source})")

                print("☁️ EmoQ Thinking...", end="\r")
                stream = ask_deepseek(text, final_emotion, identity, system_prompt)
                
                if stream:
                    print("🤖 EmoQ: ", end="")
                    full_response = "" # 用来收集完整的句子
                    for chunk in stream:
                        if chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            print(content, end='', flush=True)
                            full_response += content # 拼接句子
                    print("") 
                    
                    # 🗣️ 文字打印完后，立刻呼叫喇叭把它读出来
                    speak(full_response)
                else:
                    print("🤖 (Network Error)")
                
                gc.collect()

            except KeyboardInterrupt:
                print("\n👋 Exiting...")
                break
            except Exception as e:
                print(f"\n⚠️ Error: {e}")
                continue

if __name__ == "__main__":
    main()