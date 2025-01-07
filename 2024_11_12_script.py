import time
# time.sleep(60*1)
import subprocess
import psutil
import os

os.add_dll_directory(r"C:/opencv-4.9.0/build/install/x64/vc16/bin")
os.add_dll_directory(r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin")
import cv2

# 获取文件夹下所有子文件夹的名称
folder_path = r'D:\TianJinGangTest\22fps_crop'
subfolder_names = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
# 打印所有子文件夹名称
for subfolder_name in subfolder_names:
    print(subfolder_name)
    video = cv2.VideoCapture(os.path.join(folder_path, subfolder_name, "front.mp4"))
    # 获取视频帧数和帧率
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = video.get(cv2.CAP_PROP_FPS)
    # 计算视频时长（单位：秒）
    duration = frame_count / fps
    process1 = subprocess.Popen(['python', 'V0.2.0_2024_11_12_test.py', folder_path, subfolder_name])
    # process1 = subprocess.Popen(['python', 'V0.4.0_save_test.py', subfolder_name])
    time.sleep(duration + 10)
    # 获取当前 Python 进程的 PID
    # 获取当前脚本的进程ID
    current_pid = os.getpid()

    # 尝试杀死所有 Python 相关进程，除了当前进程
    for proc in psutil.process_iter(attrs=['pid', 'name']):
        try:
            # 检查进程名称是否是 Python 解释器且不是当前脚本进程
            if proc.info['name'].lower() in ['python.exe', 'python3.exe', 'pythonw.exe'] and proc.info['pid'] != current_pid:
                proc.terminate()  # 先尝试优雅地终止进程
                print(f"Terminated Python process with PID {proc.info['pid']}")
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    # 等待一小段时间确保进程被终止
    time.sleep(8)

    # 再次检查是否有未终止的 Python 进程，强制杀死
    for proc in psutil.process_iter(attrs=['pid', 'name']):
        try:
            if proc.info['name'].lower() in ['python.exe', 'python3.exe', 'pythonw.exe'] and proc.info['pid'] != current_pid:
                proc.kill()  # 强制杀死进程
                print(f"Killed Python process with PID {proc.info['pid']}")
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    time.sleep(8)

    print("All other Python processes have been terminated.")
