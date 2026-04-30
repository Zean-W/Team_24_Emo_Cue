from flask import Flask, Response, render_template_string
import cv2
import numpy as np
import onnxruntime as ort
import time
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
        }
        #video { 
            border: 3px solid #4CAF50; 
            max-width: 90%;
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
    <script>
        setInterval(function() {
            fetch('/emotion_status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('emotion').textContent = data.emotion + ' ' + data.emoji;
                    document.getElementById('emotion').className = 'emotion-display ' + data.class;
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

# 加载模型
print("加载模型...")
session = ort.InferenceSession("/home/fuyang/emotion_recognition/emotion-ferplus-8.onnx")
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

def gstreamer_pipeline():
    return (
        "nvarguscamerasrc sensor-id=0 ! "
        "video/x-raw(memory:NVMM), width=1280, height=720, "
        "format=NV12, framerate=30/1 ! "
        "nvvidconv flip-method=0 ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )

def generate_frames():
    global current_emotion, current_class, current_emoji
    
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    print("✅ 摄像头启动")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame_count += 1
        
        # 每3帧处理一次
        if frame_count % 3 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
            
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                
                face_roi = gray[y:y+h, x:x+w]
                face_input = cv2.resize(face_roi, (64, 64))
                face_input = face_input.astype(np.float32) / 255.0
                face_input = np.expand_dims(face_input, axis=(0, 1))
                
                outputs = session.run(None, {session.get_inputs()[0].name: face_input})
                probs = np.exp(outputs[0][0]) / np.sum(np.exp(outputs[0][0]))
                
                emotion, cls, emoji = simplify_emotion(probs)
                current_emotion = emotion
                current_class = cls
                current_emoji = emoji
                
                color = (0, 255, 0) if cls == "positive" else (0, 0, 255) if cls == "negative" else (255, 255, 0)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"{emotion} {emoji}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
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
        'emoji': current_emoji
    })

if __name__ == '__main__':
    print("\n🚀 启动表情识别服务器")
    print("📱 浏览器打开: http://10.0.0.81:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
