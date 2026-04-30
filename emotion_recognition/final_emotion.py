from flask import Flask, Response, render_template_string
import cv2
import numpy as np
import onnxruntime as ort
import json

app = Flask(__name__)

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
    </style>
</head>
<body>
    <h1>🎭 实时表情识别</h1>
    <img id="video" src="/video_feed">
    <div id="emotion" class="emotion">启动中...</div>
    <script>
        setInterval(() => {
            fetch('/status').then(r => r.json()).then(d => {
                let e = document.getElementById('emotion');
                e.textContent = d.emotion + ' ' + d.emoji;
                e.className = 'emotion ' + d.class;
            });
        }, 500);
    </script>
</body>
</html>
'''

# 全局状态
emotion_state = {"emotion": "初始化", "class": "neutral", "emoji": "🤖"}

# 加载模型
print("加载模型...")
session = ort.InferenceSession("/home/fuyang/emotion_recognition/emotion-ferplus-8.onnx")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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

def gst_pipeline():
    return (
        "nvarguscamerasrc sensor-id=0 ! "
        "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
        "nvvidconv flip-method=0 ! video/x-raw, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! appsink drop=true max-buffers=1 sync=false"
    )

def generate():
    cap = cv2.VideoCapture(gst_pipeline(), cv2.CAP_GSTREAMER)
    print("✅ 摄像头启动")
    count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        if count % 3 == 0:  # 每3帧处理一次
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            
            if len(faces) > 0:
                x, y, w, h = faces[0]
                face = cv2.resize(gray[y:y+h, x:x+w], (64, 64))
                face = face.astype(np.float32) / 255.0
                face = face.reshape(1, 1, 64, 64)
                
                out = session.run(None, {session.get_inputs()[0].name: face})
                probs = np.exp(out[0][0]) / np.sum(np.exp(out[0][0]))
                
                e, c, em = simplify(probs)
                emotion_state.update({"emotion": e, "class": c, "emoji": em})
                
                color = (0,255,0) if c=="positive" else (0,0,255) if c=="negative" else (255,255,0)
                cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
                cv2.putText(frame, f"{e} {em}", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        count += 1
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index(): return HTML

@app.route('/video_feed')
def video_feed(): return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status(): return json.dumps(emotion_state)

if __name__ == '__main__':
    print("\n🚀 启动服务器...")
    print("📱 浏览器打开: http://10.0.0.81:5000\n")
    app.run(host='0.0.0.0', port=5000)
