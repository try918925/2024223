import torch

if torch.cuda.is_available():
    print(f"PyTorch 找到了 {torch.cuda.device_count()} 个 GPU")
    print(f"当前使用的 GPU 是: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("没有可用的 GPU")



import paddle

# 检查 PaddlePaddle 是否支持 GPU
if paddle.is_compiled_with_cuda():
    print("PaddlePaddle 支持 GPU")
else:
    print("PaddlePaddle 不支持 GPU")

# 检查 PaddlePaddle 是否能找到 GPU
gpu_available = paddle.device.is_compiled_with_cuda()
print(f"GPU 是否可用: {gpu_available}")



import tensorrt
print(tensorrt.__version__)

