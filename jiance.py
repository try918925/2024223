
import psutil
import GPUtil
import os
import time

def get_cpu_memory_usage():
    process = psutil.Process(os.getpid())
    cpu_percent = process.cpu_percent(interval=1)
    memory_info = process.memory_info()
    memory_percent = process.memory_percent()
    memory_rss = memory_info.rss
    memory_vms = memory_info.vms
    return cpu_percent, memory_percent, memory_rss, memory_vms

def get_gpu_usage():
    gpus = GPUtil.getGPUs()
    gpu_info = []
    for gpu in gpus:
        gpu_info.append({
            'id': gpu.id,
            'name': gpu.name,
            'load': gpu.load * 100,
            'memoryUsed': gpu.memoryUsed,
            'memoryTotal': gpu.memoryTotal,
            'memoryUtil': gpu.memoryUtil * 100,
            'temperature': gpu.temperature,
        })
    return gpu_info

if __name__ == "__main__":
    while True:
        cpu_percent, memory_percent, memory_rss, memory_vms = get_cpu_memory_usage()
        print(f"CPU Usage: {cpu_percent}%")
        print(f"Memory Usage: {memory_percent}%")
        print(f"Memory RSS: {memory_rss / (1024 * 1024)} MB")
        print(f"Memory VMS: {memory_vms / (1024 * 1024)} MB")

        gpu_info = get_gpu_usage()
        for gpu in gpu_info:
            print(f"GPU ID: {gpu['id']}")
            print(f"GPU Name: {gpu['name']}")
            print(f"GPU Load: {gpu['load']}%")
            print(f"GPU Memory Used: {gpu['memoryUsed']} MB")
            print(f"GPU Memory Total: {gpu['memoryTotal']} MB")
            print(f"GPU Memory Util: {gpu['memoryUtil']}%")
            print(f"GPU Temperature: {gpu['temperature']} C")

        time.sleep(5)  # 每5秒打印一次
