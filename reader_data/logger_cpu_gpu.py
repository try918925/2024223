import time
import psutil
import platform


class cpu_info(object):
    '''
        get cpu information, base on psutil.
    '''

    def __init__(self):
        self.mem = None
        self.memory_total = None
        self.memory_free = None
        self.memory_percent = None
        self.memory_used = None
        self.cpu_used_percent_details = None
        self.cpu_core_nums = None
        self.timestamp = None
        self.cpu_used_percent = None

    def __call__(self, ):
        self.mem = psutil.virtual_memory()
        self.memory_total = self.mem.total  # 内存总量, bytes
        self.memory_free = self.mem.free  # 空闲内存总量
        self.memory_percent = self.mem.percent  # 内存使用百分比
        self.memory_used = self.mem.used  # 内存使用量
        self.cpu_used_percent_details = psutil.cpu_percent(percpu=True)  # 每个核的使用率
        self.cpu_core_nums = psutil.cpu_count(logical=False)  # cpu 核心数
        self.timestamp = round(time.time() * 1000)  # 信息时间戳
        self.cpu_used_percent = psutil.cpu_percent(interval=None, percpu=False)  # cpu平均利用率

        self.temperature = None  # for linux psutil.sensors_temperatures()

    def get_info(self):
        self.mem = psutil.virtual_memory()
        self.memory_total = round(self.mem.total / 1024.0 / 1024 / 1024, 2)
        self.memory_free = self.mem.free
        self.memory_percent = self.mem.percent
        self.memory_used = self.mem.used
        self.cpu_used_percent_details = psutil.cpu_percent(percpu=True)
        self.cpu_core_nums = psutil.cpu_count(logical=False)
        self.timestamp = round(time.time() * 1000)
        self.cpu_used_percent = psutil.cpu_percent(interval=None, percpu=False)
        self.temperature = None  # for linux psutil.sensors_temperatures()

        return (self.timestamp, self.temperature,
                self.memory_total, self.memory_percent,
                self.cpu_core_nums, self.cpu_used_percent)


from subprocess import Popen, PIPE
from distutils import spawn
import os


def safeFloatCast(strNumber):
    try:
        number = float(strNumber)
    except ValueError:
        number = float('nan')
    return number


class GPU:
    def __init__(self, ID, uuid, load, memoryTotal, memoryUsed, memoryFree, driver, gpu_name, serial, display_mode,
                 display_active, temp_gpu, timestamp):
        self.id = ID  # gpu 序号
        self.uuid = uuid
        self.load = load  # gpu利用率
        self.memoryUtil = round(float(memoryUsed) / float(memoryTotal), 3)  # 显存利用率
        self.memoryTotal = round(memoryTotal / 1024, 2)  # 总显存数， GB
        self.memoryUsed = memoryUsed  # 显存利用率
        self.memoryFree = memoryFree  # 空闲显存
        self.driver = driver  # 驱动版本号
        self.name = gpu_name
        self.serial = serial
        self.display_mode = display_mode
        self.display_active = display_active
        self.temperature = temp_gpu  # gpu 温度
        self.timestamp = timestamp


class gpu_info(object):
    '''
        get gpu information based on nvidia-smi command. 
    '''

    def __init__(self):
        if platform.system() == "Windows":
            # If the platform is Windows and nvidia-smi 
            # could not be found from the environment path, 
            # try to find it from system drive with default installation path
            nvidia_smi = spawn.find_executable('nvidia-smi')
            if nvidia_smi is None:
                nvidia_smi = "%s\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe" % os.environ['systemdrive']
        else:
            nvidia_smi = "nvidia-smi"
        self.cmd = nvidia_smi
        self.GPUs = []

    def __call__(self, ):
        pass

    def get_info(self):
        self.timestamp = round(time.time() * 1000)

        try:
            p = Popen([self.cmd,
                       "--query-gpu=index,uuid,utilization.gpu,memory.total,memory.used,memory.free,driver_version,name,gpu_serial,display_active,display_mode,temperature.gpu",
                       "--format=csv,noheader,nounits"], stdout=PIPE)
            stdout, stderror = p.communicate()
        except:
            return []
        output = stdout.decode('UTF-8')
        lines = output.split(os.linesep)
        numDevices = len(lines) - 1
        self.GPUs = []
        for g in range(numDevices):
            line = lines[g]
            # print(line)
            vals = line.split(', ')
            # print(vals)
            for i in range(12):
                # print(vals[i])
                if (i == 0):
                    deviceIds = int(vals[i])
                elif (i == 1):
                    uuid = vals[i]
                elif (i == 2):
                    gpuUtil = safeFloatCast(vals[i])
                elif (i == 3):
                    memTotal = safeFloatCast(vals[i])
                elif (i == 4):
                    memUsed = safeFloatCast(vals[i])
                elif (i == 5):
                    memFree = safeFloatCast(vals[i])
                elif (i == 6):
                    driver = vals[i]
                elif (i == 7):
                    gpu_name = vals[i]
                elif (i == 8):
                    serial = vals[i]
                elif (i == 9):
                    display_active = vals[i]
                elif (i == 10):
                    display_mode = vals[i]
                elif (i == 11):
                    temp_gpu = safeFloatCast(vals[i])
            self.GPUs.append(
                GPU(deviceIds, uuid, gpuUtil, memTotal, memUsed, memFree, driver, gpu_name, serial, display_mode,
                    display_active, temp_gpu, self.timestamp))
        # print(self.GPUs[0].id)
        info = []
        for item in self.GPUs:
            info.append((item.id, item.timestamp, item.load, item.memoryTotal, item.memoryUtil, item.temperature,))
        return info


if __name__ == '__main__':
    c_info = cpu_info()
    g_info = gpu_info()

    for i in range(1000):
        print("cpu:", c_info.get_info())
        print("gpu:", g_info.get_info())
        time.sleep(0.5)
