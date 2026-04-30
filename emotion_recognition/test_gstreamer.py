import cv2

print("测试GStreamer pipeline...")

pipeline = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=1920, height=1080, format=NV12, framerate=30/1 ! "
    "nvvidconv ! "
    "video/x-raw, width=960, height=540, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! appsink"
)

print(f"Pipeline: {pipeline}")

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if cap.isOpened():
    print("✅ 摄像头打开成功")
    for i in range(5):
        ret, frame = cap.read()
        if ret:
            print(f"✅ 成功读取帧 {i+1}, shape: {frame.shape}")
            if i == 0:
                cv2.imwrite("test.jpg", frame)
                print("已保存test.jpg")
        else:
            print(f"❌ 无法读取帧 {i+1}")
    cap.release()
else:
    print("❌ 无法打开摄像头")
