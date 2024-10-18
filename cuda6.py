import multiprocessing
from functools import partial
import pycuda.driver as cuda

def init_pycuda(process_id):
    try:
        cuda.init()
        device = cuda.Device(0)  # 使用设备 0
        context = device.make_context()
        # 在此添加 PyCUDA 初始化代码
        print("PyCUDA initialized in process", process_id)
    except Exception as e:
        print("Error initializing PyCUDA:", e)

def main():
    # 创建进程池
    num_processes = 4  # 例如，使用4个进程
    pool = multiprocessing.Pool(processes=num_processes)
    # 使用 functools.partial 部分应用参数
    init_pycuda_partial = partial(init_pycuda, 0)  # 传递参数 0
    # 在每个进程中初始化 PyCUDA
    pool.map(init_pycuda_partial, range(num_processes))
    # 关闭进程池
    pool.close()
    pool.join()

if __name__ == "__main__":
    main()
