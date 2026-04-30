import cv2
import numpy as np
import subprocess
import os

print("基础摄像头测试")
print("="*50)

# 检查服务状态
print("\n1. 检查nvargus-daemon状态:")
result = subprocess.run(["systemctl", "is-active", "nvargus-daemon"], capture_output=True, text=True)
print(f"   nvargus-daemon: {result.stdout.strip()}")

# 检查设备权限
print("\n2. 检查/dev/video0权限:")
if os.path.exists("/dev/video0"):
    import stat
    st = os.stat("/dev/video0")
    print(f"   权限: {oct(st.st_mode)[-3:]}")
    print(f"   可读: {os.access('/dev/video0', os.R_OK)}")
    print(f"   可写: {os.access('/dev/video0', os.W_OK)}")

# 尝试最简单的V4L2访问
print("\n3. 尝试直接V4L2访问:")
cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

if cap.isOpened():
    print("   ✅ 打开成功")
    # 尝试多次读取
    for i in range(5):
        ret, frame = cap.read()
        if ret:
            print(f"   ✅ 第{i+1}次读取成功: {frame.shape}")
            if i == 0:
                cv2.imwrite("v4l2_test.jpg", frame)
            break
        else:
            print(f"   ⏳ 第{i+1}次读取失败，继续尝试...")
            import time
            time.sleep(0.5)
    cap.release()
else:
    print("   ❌ 无法打开")

# 创建一个假图像用于测试Web服务器
print("\n4. 创建测试图像:")
test_img = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.putText(test_img, "Camera Test", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imwrite("fake_camera.jpg", test_img)
print("   已创建fake_camera.jpg用于测试")
