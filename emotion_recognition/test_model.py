import cv2
import numpy as np
import onnxruntime as ort
import os

# 检查模型文件
model_path = "emotion-ferplus-8.onnx"
if not os.path.exists(model_path):
    print(f"❌ 模型文件不存在: {model_path}")
    exit()

print(f"✅ 模型文件大小: {os.path.getsize(model_path)/1024/1024:.2f} MB")

try:
    session = ort.InferenceSession(model_path)
    print("✅ 模型加载成功!")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    exit()

print("输入形状:", session.get_inputs()[0].shape)
print("输出形状:", session.get_outputs()[0].shape)

# 创建测试输入
dummy_input = np.random.randn(1, 1, 64, 64).astype(np.float32)
outputs = session.run(None, {session.get_inputs()[0].name: dummy_input})
print("✅ 推理测试成功! 输出shape:", outputs[0].shape)

# 表情类别
emotions = ['neutral', 'happiness', 'surprise', 'sadness', 
            'anger', 'disgust', 'fear', 'contempt']
print("\n支持的8种表情:")
for i, emo in enumerate(emotions):
    print(f"  {i}: {emo}")

# 测试softmax输出
probs = np.exp(outputs[0][0]) / np.sum(np.exp(outputs[0][0]))
print("\n测试输出概率分布（随机输入）:")
for emo, prob in zip(emotions, probs):
    print(f"  {emo}: {prob:.2%}")
    
print("\n✅ 所有测试通过！模型可以使用。")
