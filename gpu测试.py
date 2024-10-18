# -*- coding: utf-8 -*-
import os
import time
import gc

os.add_dll_directory(r"C:/opencv_build_data/opencv-4.9.0/build/install/x64/vc16/bin")
os.add_dll_directory(r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin")
import cv2
import cv2.cuda
import time

# 读取图片
image = cv2.imread(r'C:/Users/Install/Desktop/2024223/my_test_img/top/2024_09_21_19_02_50.jpg')

# 使用CPU进行缩放
start_time = time.time()
cpu_resized = cv2.resize(image, (900, 900), interpolation=cv2.INTER_LINEAR)
cpu_time = time.time() - start_time
print(f"CPU resize time: {cpu_time:.4f} seconds")


start_time = time.time()
# 使用GPU进行缩放
gpu_image = cv2.cuda_GpuMat()
gpu_image.upload(image)  # 上传图像到GPU

gpu_resized = cv2.cuda.resize(gpu_image, (900, 900), interpolation=cv2.INTER_LINEAR)
gpu_time = time.time() - start_time
# 下载GPU图像到CPU以进行对比
gpu_resized_cpu = gpu_resized.download()
print(f"GPU resize time: {gpu_time:.4f} seconds")


# 显示对比结果
cv2.imshow("CPU Resized Image", cpu_resized)
cv2.imshow("GPU Resized Image", gpu_resized_cpu)
cv2.waitKey(0)
cv2.destroyAllWindows()
