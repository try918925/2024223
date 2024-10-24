import psutil
import GPUtil
import time


def main():
    cpu_list = []
    memory_list = []
    gpu_list = []
    start_time = time.time()
    while True:
        # end_time = time.time() - start_time
        # start_time = time.time()
        cpu_usage = psutil.cpu_percent(interval=1)
        cpu_list.append(cpu_usage)

        # 获取内存利用率
        memory_info = psutil.virtual_memory()
        memory_usage = memory_info.percent
        memory_list.append(memory_usage)

        # 获取 GPU 利用率
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            # print(f"GPU ID: {gpu.id}, 利用率: {gpu.load * 100:.2f}%, 内存利用率: {gpu.memoryUtil * 100:.2f}%")
            gpu_list.append(round(gpu.load * 100))
        if len(cpu_list) > 25:
            cpu_max = max(cpu_list)
            memory_max = max(memory_list)
            gpu_max = max(gpu_list)
            cpu_list.clear()
            memory_list.clear()
            gpu_list.clear()
            print(f"cpu:{cpu_max} gpu:{gpu_max} 内存:{memory_max}")


if __name__ == '__main__':
    main()
