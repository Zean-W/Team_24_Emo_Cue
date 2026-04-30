import cv2
import numpy as np

print("测试IMX477摄像头...")

# 简单的GStreamer管道
def get_camera():
    # 先试试最简单的
    return cv2.VideoCapture(0)

cap = get_camera()
if not cap.isOpened():
    print("❌ 方法1失败，尝试GStreamer...")
    # 尝试GStreamer
    pipeline = "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=1920, height=1080, format=NV12, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if cap.isOpened():
    print("✅ 摄像头打开成功!")
    ret, frame = cap.read()
    if ret:
        print(f"✅ 成功读取一帧: {frame.shape}")
        cv2.imwrite("test_frame.jpg", frame)
        print("✅ 保存测试图片: test_frame.jpg")
    else:
        print("❌ 无法读取帧")
    cap.release()
else:
    print("❌ 无法打开摄像头")
