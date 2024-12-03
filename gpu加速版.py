import torch

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建一个随机的张量，并将其移动到 GPU
tensor = torch.randn(10000, 10000).to(device)

# 在 GPU 上进行计算
result = torch.matmul(tensor, tensor.T)
print(result)

print(result.is_cuda)

a = result.cpu()


print(a.is_cuda)