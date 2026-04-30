import cv2
import subprocess

print("测试IMX477专用pipeline...")

# 先检查摄像头状态
print("\n检查摄像头信息:")
subprocess.run(["v4l2-ctl", "--list-devices"], capture_output=False)

# IMX477专用的pipeline
pipelines = [
    # 选项1: 最基础的
    "nvarguscamerasrc sensor-id=0 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink",
    
    # 选项2: 指定分辨率
    "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=21/1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink",
    
    # 选项3: 更低分辨率
    "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=640, height=480, format=NV12, framerate=21/1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink",
    
    # 选项4: 不指定framerate
    "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=1280, height=720, format=NV12 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"
]

for i, pipeline in enumerate(pipelines):
    print(f"\n测试Pipeline {i+1}:")
    print(f"  {pipeline[:80]}...")
    
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"  ✅ 成功! Shape: {frame.shape}")
            cv2.imwrite(f"test_pipeline_{i+1}.jpg", frame)
            print(f"  已保存 test_pipeline_{i+1}.jpg")
            cap.release()
            print("\n🎉 找到工作的pipeline！使用Pipeline", i+1)
            break
        else:
            print(f"  ❌ 打开但无法读取")
        cap.release()
    else:
        print(f"  ❌ 无法打开")
else:
    print("\n所有pipeline都失败了")
    
# 如果都失败，尝试直接测试
print("\n直接测试nvgstcapture能否工作:")
print("运行: nvgstcapture-1.0 --mode=1 --sensor-id=0")
print("(这会打开一个窗口，SSH看不到，但可以测试摄像头)")
