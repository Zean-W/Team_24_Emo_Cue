from flask import Flask, Response, render_template_string
import cv2
import numpy as np
import onnxruntime as ort
import time
from datetime import datetime
import json

app = Flask(__name__)

# HTML模板
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Jetson Emotion Recognition</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            text-align: center; 
            background: #1a1a1a;
            color: white;
            margin: 0;
            padding: 20px;
        }
        #video { 
            border: 3px solid #4CAF50; 
            max-width: 90%;
            height: auto;
            border-radius: 10px;
        }
        .emotion-display {
            font-size: 48px;
            margin: 20px auto;
            padding: 20px;
            border-radius: 10px;
            background: #333;
            max-width: 500px;
        }
        .positive { background: #4CAF50 !important; }
        .negative { background: #f44336 !important; }
        .neutral { background: #ff9800 !important; }
        .stats {
            margin: 20px auto;
            padding: 15px;
            background: #333;
            border-radius: 5px;
            max-width: 600px;
        }
        h1 { color: #4CAF50; }
    </style>
</head>
<body>
    <h1>🎭 实时表情识别 - Jetson Orin Nano</h1>
    <img id="video" src="/video_feed">
    <div id="emotion" class="emotion-display">初始化中...</div>
    <div class="stats">
        <p>📷 IMX477 (12.3MP) | 🧠 Emotion-FERPlus-8 | 💻 ONNX Runtime</p>
        <p id="fps">FPS: 计算中...</p>
        <p id="details">详细: 等待检测...</p>
    </div>
    <script>
        setInterval(function() {
            fetch('/emotion_status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('emotion').textContent = data.emotion + ' ' + data.emoji;
                    document.getElementById('emotion').className = 'emotion-display ' + data.class;
                    document.getElementById('fps').textContent = 'FPS: ' + data.fps.toFixed(1);
                    document.getElementById('details').textContent = '详细: ' + data.details;
                });
        }, 500);
    </script>
</body>
</html>
'''

# 全局变量
current_emotion = "初始化"
current_class = "neutral"
current_emoji = "🤖"
current_fps = 0
current_details = "等待..."

# 加载模型
print("加载表情识别模型...")
session = ort.InferenceSession("emotion-ferplus-8.onnx")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
print("✅ 模型加载完成")

emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']

def simplify_emotion(probs):
    positive = probs[1] + probs[2]
    negative = probs[3] + probs[4] + probs[5] + probs[6]
    neutral = probs[0] + probs[7]
    
    if positive > max(negative, neutral):
        return "Positive", "positive", "😊"
    elif negative > max(positive, neutral):
        return "Negative", "negative", "😔"
    else:
        return "Neutral", "neutral", "😐"

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0
):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
    )

def generate_frames():
    global current_emotion, current_class, current_emoji, current_fps, current_details
    
    print("启动摄像头...")
    # 使用GStreamer pipeline
    pipeline = gstreamer_pipeline()
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return
    
    print("✅ 摄像头启动成功")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 读取帧失败，重试...")
            continue
        
        frame_count += 1
        
        # 计算FPS
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            current_fps = 30 / elapsed if elapsed > 0 else 0
            start_time = time.time()
        
        # 每2帧处理一次（提高响应速度）
        if frame_count % 2 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
            
            if len(faces) > 0:
                # 只处理最大的脸
                faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
                (x, y, w, h) = faces[0]
                
                # 提取并预处理人脸
                face_roi = gray[y:y+h, x:x+w]
                face_input = cv2.resize(face_roi, (64, 64))
                face_input = face_input.astype(np.float32) / 255.0
                face_input = np.expand_dims(face_input, axis=0)
                face_input = np.expand_dims(face_input, axis=0)
                
                # 推理
                outputs = session.run(None, {session.get_inputs()[0].name: face_input})
                probs = np.exp(outputs[0][0]) / np.sum(np.exp(outputs[0][0]))
                
                # 获取top情绪
                max_idx = np.argmax(probs)
                max_emotion = emotions[max_idx]
                confidence = probs[max_idx]
                
                # 简化分类
                emotion, cls, emoji = simplify_emotion(probs)
                current_emotion = emotion
                current_class = cls
                current_emoji = emoji
                current_details = f"{max_emotion} ({confidence:.1%})"
                
                # 画框和标签
                color = (0, 255, 0) if cls == "positive" else (0, 0, 255) if cls == "negative" else (0, 165, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                
                # 添加标签背景
                label = f"{emotion} {emoji}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                cv2.rectangle(frame, (x, y-35), (x+label_size[0]+10, y), color, -1)
                cv2.putText(frame, label, (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            else:
                current_emotion = "No Face"
                current_emoji = "👤"
                current_details = "未检测到人脸"
        
        # 添加FPS显示
        cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 编码为JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ret:
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/emotion_status')
def emotion_status():
    return json.dumps({
        'emotion': current_emotion,
        'class': current_class,
        'emoji': current_emoji,
        'fps': current_fps,
        'details': current_details
    })

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Web服务器启动中...")
    print("="*50)
    
    # 获取IP地址
    import subprocess
    ip = subprocess.check_output(['hostname', '-I']).decode().split()[0]
    
    print(f"\n✅ 服务器已启动!")
    print(f"📱 在Mac浏览器打开: http://{ip}:5000")
    print(f"🔄 或者尝试: http://localhost:5000 (如果在Jetson本地)")
    print("\n按 Ctrl+C 停止服务器\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
