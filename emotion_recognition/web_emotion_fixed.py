from flask import Flask, Response, render_template_string
import cv2
import numpy as np
import onnxruntime as ort
import time
import json
import subprocess
import os

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
    </style>
</head>
<body>
    <h1>🎭 实时表情识别 - Jetson Orin Nano</h1>
    <img id="video" src="/video_feed">
    <div id="emotion" class="emotion-display">初始化...</div>
    <div class="stats">
        <p>📷 IMX477 | 🧠 Emotion-FERPlus-8 | 💻 ONNX Runtime</p>
        <p id="details">等待检测...</p>
    </div>
    <script>
        setInterval(function() {
            fetch('/emotion_status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('emotion').textContent = data.emotion + ' ' + data.emoji;
                    document.getElementById('emotion').className = 'emotion-display ' + data.class;
                    document.getElementById('details').textContent = data.details;
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
current_details = "启动中..."

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

def get_camera_pipeline():
    """获取正确的摄像头pipeline"""
    # 使用nvgstcapture相同的配置
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)30/1 ! "
        "nvvidconv ! "
        "video/x-raw, width=(int)640, height=(int)360, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! "
        "appsink drop=1 max-buffers=2"
    )

def capture_frame_nvgst():
    """使用nvgstcapture抓取一帧"""
    # 创建临时文件路径
    temp_file = "/tmp/capture.jpg"
    
    # 使用nvgstcapture抓取一张图片
    cmd = f"timeout 2 nvgstcapture-1.0 --automate --capture --image-res=3 --file-name={temp_file}"
    subprocess.run(cmd, shell=True, capture_output=True)
    
    if os.path.exists(temp_file + "_0.jpg"):
        frame = cv2.imread(temp_file + "_0.jpg")
        os.remove(temp_file + "_0.jpg")
        return True, frame
    return False, None

def generate_frames():
    global current_emotion, current_class, current_emoji, current_details
    
    print("尝试打开摄像头...")
    
    # 先尝试GStreamer
    pipeline = get_camera_pipeline()
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    
    use_nvgst_fallback = False
    if not cap.isOpened():
        print("⚠️ GStreamer失败，使用nvgstcapture fallback模式")
        use_nvgst_fallback = True
    else:
        print("✅ GStreamer摄像头打开成功")
    
    frame_count = 0
    last_frame = None
    
    while True:
        if use_nvgst_fallback:
            # 使用nvgstcapture抓取（较慢但稳定）
            if frame_count % 30 == 0:  # 每30帧抓一次
                ret, frame = capture_frame_nvgst()
                if ret:
                    last_frame = frame
            else:
                frame = last_frame if last_frame is not None else np.zeros((720, 1280, 3), dtype=np.uint8)
                ret = last_frame is not None
        else:
            # 正常的VideoCapture读取
            ret, frame = cap.read()
        
        if not ret:
            # 创建占位符图像
            frame = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Camera Error", (200, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 调整大小以提高性能
        if frame.shape[0] > 360:
            frame = cv2.resize(frame, (640, 360))
        
        frame_count += 1
        
        # 每3帧处理一次表情
        if frame_count % 3 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
            
            if len(faces) > 0:
                # 处理最大的脸
                faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
                (x, y, w, h) = faces[0]
                
                # 提取人脸
                face_roi = gray[y:y+h, x:x+w]
                face_input = cv2.resize(face_roi, (64, 64))
                face_input = face_input.astype(np.float32) / 255.0
                face_input = np.expand_dims(face_input, axis=0)
                face_input = np.expand_dims(face_input, axis=0)
                
                # 推理
                outputs = session.run(None, {session.get_inputs()[0].name: face_input})
                probs = np.exp(outputs[0][0]) / np.sum(np.exp(outputs[0][0]))
                
                # 获取结果
                max_idx = np.argmax(probs)
                max_emotion = emotions[max_idx]
                confidence = probs[max_idx]
                
                emotion, cls, emoji = simplify_emotion(probs)
                current_emotion = emotion
                current_class = cls
                current_emoji = emoji
                current_details = f"{max_emotion} ({confidence:.1%})"
                
                # 画框
                color = (0, 255, 0) if cls == "positive" else (0, 0, 255) if cls == "negative" else (0, 165, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"{emotion} {emoji}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            else:
                current_emotion = "No Face"
                current_emoji = "👤"
                current_details = "未检测到人脸"
        
        # 编码
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

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
        'details': current_details
    })

if __name__ == '__main__':
    ip = subprocess.check_output(['hostname', '-I']).decode().split()[0]
    
    print("\n" + "="*50)
    print("🚀 表情识别服务器启动")
    print("="*50)
    print(f"📱 浏览器打开: http://{ip}:5000")
    print("\n按 Ctrl+C 停止\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
