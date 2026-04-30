from flask import Flask, Response, render_template_string
import cv2
import numpy as np
import onnxruntime as ort
import json
import os
import time

app = Flask(__name__)

# ============ 检测间隔配置 ============
DETECTION_INTERVAL = 2.0  # 默认2秒检测一次
DETECTION_MODE = "default"  # 可选: "demo", "default", "idle"

# 预设模式
MODES = {
    "demo": {"interval": 0.5, "smooth_window": 3},     # 演示模式：0.5秒
    "default": {"interval": 2.0, "smooth_window": 3},  # 默认模式：2秒
    "idle": {"interval": 5.0, "smooth_window": 5}      # 省电模式：5秒
}

# 选择模式
mode_config = MODES[DETECTION_MODE]
DETECTION_INTERVAL = mode_config["interval"]
SMOOTH_WINDOW = mode_config["smooth_window"]

HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>表情识别 - Jetson</title>
    <style>
        body { background: #1a1a1a; color: white; text-align: center; font-family: Arial; }
        #video { border: 3px solid #4CAF50; max-width: 80%; border-radius: 10px; }
        .emotion { font-size: 48px; margin: 20px; padding: 20px; background: #333; border-radius: 10px; }
        .positive { background: #4CAF50 !important; }
        .negative { background: #f44336 !important; }
        .neutral { background: #ff9800 !important; }
        .info { font-size: 16px; margin: 10px; color: #888; }
    </style>
</head>
<body>
    <h1>🎭 实时表情识别</h1>
    <img id="video" src="/video_feed">
    <div id="emotion" class="emotion">启动中...</div>
    <div class="info">
        模式: <span id="mode">-</span> | 
        检测间隔: <span id="interval">-</span>秒 | 
        FPS: <span id="fps">-</span>
    </div>
    <script>
        setInterval(() => {
            fetch('/status').then(r => r.json()).then(d => {
                document.getElementById('emotion').textContent = d.emotion + ' ' + d.emoji;
                document.getElementById('emotion').className = 'emotion ' + d.class;
                document.getElementById('mode').textContent = d.mode;
                document.getElementById('interval').textContent = d.interval;
                document.getElementById('fps').textContent = d.fps;
            });
        }, 500);
    </script>
</body>
</html>
'''

# 全局状态
emotion_state = {
    "emotion": "初始化", 
    "class": "neutral", 
    "emoji": "🤖",
    "mode": DETECTION_MODE,
    "interval": DETECTION_INTERVAL,
    "fps": 0
}

# 检测历史（用于平滑）
emotion_history = []

# 加载模型
print(f"加载模型... 模式: {DETECTION_MODE}, 间隔: {DETECTION_INTERVAL}秒")
session = ort.InferenceSession(
    "/home/fuyang/emotion_recognition/emotion-ferplus-8.onnx",
    providers=["CPUExecutionProvider"]
)

# 找到正确的cascade路径
CASCADE_PATH = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
if not os.path.exists(CASCADE_PATH):
    CASCADE_PATH = "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml"
    if not os.path.exists(CASCADE_PATH):
        print("❌ 找不到haarcascade文件，下载一个...")
        import urllib.request
        CASCADE_PATH = "/tmp/haarcascade_frontalface_default.xml"
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        urllib.request.urlretrieve(url, CASCADE_PATH)
        print("✅ 下载完成")

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    print("❌ 无法加载人脸检测模型")
    exit()

print("✅ 模型加载完成")

emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']

def simplify(probs):
    pos = probs[1] + probs[2]
    neg = probs[3] + probs[4] + probs[5] + probs[6]
    neu = probs[0] + probs[7]
    if pos > max(neg, neu):
        return "Positive", "positive", "😊"
    elif neg > max(pos, neu):
        return "Negative", "negative", "😔"
    return "Neutral", "neutral", "😐"

def smooth_emotion(history):
    """简单的多数投票平滑"""
    if not history:
        return "Neutral", "neutral", "😐"
    
    # 统计每种情绪出现次数
    emotion_counts = {}
    for e, c, em in history:
        if e not in emotion_counts:
            emotion_counts[e] = {"count": 0, "class": c, "emoji": em}
        emotion_counts[e]["count"] += 1
    
    # 找出最多的
    max_emotion = max(emotion_counts.items(), key=lambda x: x[1]["count"])
    return max_emotion[0], max_emotion[1]["class"], max_emotion[1]["emoji"]

def gst_pipeline():
    return (
        "nvarguscamerasrc sensor-id=0 ! "
        "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
        "nvvidconv flip-method=0 ! video/x-raw, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! appsink drop=true max-buffers=1 sync=false"
    )

def generate():
    global emotion_history
    
    cap = cv2.VideoCapture(gst_pipeline(), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        while True:
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(img, "Camera Error", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            _, buffer = cv2.imencode('.jpg', img)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    print("✅ 摄像头启动")
    
    last_detect_time = 0
    last_emotion = ("Neutral", "neutral", "😐")
    frame_count = 0
    fps_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame_count += 1
        now = time.time()
        
        # 计算FPS
        if now - fps_time >= 1.0:
            emotion_state["fps"] = round(frame_count / (now - fps_time), 1)
            frame_count = 0
            fps_time = now
        
        # 检查是否到了检测时间
        if now - last_detect_time >= DETECTION_INTERVAL:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
            
            if len(faces) > 0:
                x, y, w, h = faces[0]
                face = cv2.resize(gray[y:y+h, x:x+w], (64, 64))
                face = face.astype(np.float32) / 255.0
                face = face.reshape(1, 1, 64, 64)
                
                out = session.run(None, {session.get_inputs()[0].name: face})
                probs = np.exp(out[0][0]) / np.sum(np.exp(out[0][0]))
                
                current_emotion = simplify(probs)
                
                # 添加到历史并平滑
                emotion_history.append(current_emotion)
                if len(emotion_history) > SMOOTH_WINDOW:
                    emotion_history.pop(0)
                
                e, c, em = smooth_emotion(emotion_history)
                last_emotion = (e, c, em)
                emotion_state.update({"emotion": e, "class": c, "emoji": em})
                
                # 画框（使用检测到的位置）
                color = (0,255,0) if c=="positive" else (0,0,255) if c=="negative" else (255,255,0)
                cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
                cv2.putText(frame, f"{e} {em}", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                # 更新检测时间
                last_detect_time = now
            else:
                emotion_state.update({"emotion": "No Face", "class": "neutral", "emoji": "👤"})
                emotion_history = []  # 清空历史
        else:
            # 使用上次的结果画框
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
            if len(faces) > 0:
                x, y, w, h = faces[0]
                e, c, em = last_emotion
                color = (0,255,0) if c=="positive" else (0,0,255) if c=="negative" else (255,255,0)
                cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
                cv2.putText(frame, f"{e} {em}", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index(): return HTML

@app.route('/video_feed')
def video_feed(): return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status(): return json.dumps(emotion_state)

if __name__ == '__main__':
    print("\n🚀 启动服务器...")
    print(f"📱 模式: {DETECTION_MODE} (间隔: {DETECTION_INTERVAL}秒)")
    print("📱 浏览器打开: http://10.0.0.81:5000")
    print("按 Ctrl+C 停止\n")
    app.run(host='0.0.0.0', port=5000)
