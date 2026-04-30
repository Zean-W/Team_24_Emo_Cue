import cv2
import os

print("="*50)
print("测试所有可能的摄像头访问方式")
print("="*50)

# 方法1: 直接使用video索引
print("\n方法1: 测试 /dev/video0")
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print(f"✅ video0 工作! Shape: {frame.shape}")
        cv2.imwrite("test_v4l2.jpg", frame)
    else:
        print("❌ video0 打开但无法读取")
    cap.release()
else:
    print("❌ 无法打开 video0")

# 方法2: 简化的GStreamer
print("\n方法2: 简化GStreamer pipeline")
simple_pipe = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"
cap = cv2.VideoCapture(simple_pipe, cv2.CAP_GSTREAMER)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print(f"✅ GStreamer 工作! Shape: {frame.shape}")
        cv2.imwrite("test_gst.jpg", frame)
    else:
        print("❌ GStreamer 打开但无法读取")
    cap.release()
else:
    print("❌ 无法用GStreamer打开")

# 方法3: 使用nvgstcapture测试
print("\n方法3: 检查nvgstcapture")
import subprocess
result = subprocess.run(["which", "nvgstcapture-1.0"], capture_output=True, text=True)
if result.returncode == 0:
    print(f"✅ nvgstcapture-1.0 存在: {result.stdout.strip()}")
else:
    print("❌ nvgstcapture-1.0 不存在")

# 列出所有video设备
print("\n可用的video设备:")
for i in range(5):
    if os.path.exists(f"/dev/video{i}"):
        print(f"  /dev/video{i} 存在")
