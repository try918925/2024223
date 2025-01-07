# -*- coding: utf-8 -*-
import os
from multiprocessing import Process

os.add_dll_directory(r"C:/opencv-4.9.0/build/install/x64/vc16/bin")
os.add_dll_directory(r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin")
import cv2
import ctypes
from moviepy.video.io.VideoFileClip import VideoFileClip
import os
import test_infer as trt_infer
import Container_det_trt_yolov5 as cont_trt_infer
import time
import threading
from datetime import datetime
import shutil

# 设置当前目录为工作目录且与
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
execution_path = CURRENT_DIR.replace("\\", "/")
print("execution_path:", execution_path)
global_data = {
    "front": [],
    "rear": [],
    "top": [],
    "right": [],
    "left": [],
}


def filter_adjacent(global_data):
    filtered_data = {}

    for key, values in global_data.items():
        if not values:  # 如果列表为空，直接保留空列表
            filtered_data[key] = []
            continue

        # 对于 rear 和 front，阈值为 50；其他列表不变
        if key in ['rear', 'front']:
            threshold = 50
            filtered_values = [values[0]]  # 保留第一个元素
            for i in range(1, len(values)):
                # 比较当前值和最后一个保留的值的差
                if abs(values[i] - filtered_values[-1]) > threshold:
                    filtered_values.append(values[i])  # 保留当前值
            filtered_data[key] = filtered_values  # 存储过滤后的结果
        else:  # 对于 right, left 和 top 列表不变
            filtered_data[key] = values

    return filtered_data


def get_camera_info(filed_path, filed_name):
    camera_config = {
        "params_left": {
            "file_path": f"{filed_path}/{filed_name}/left.mp4",
            "roi": [400, 1300, 50, 950],
            "id": 0,
            "queue": "",
            "direction": "left"
        },
        "params_right": {
            "file_path": f"{filed_path}/{filed_name}/right.mp4",
            "roi": [400, 1300, 1650, 2550],
            "id": 2,
            "queue": "",
            "direction": "right"
        },
        "params_top": {
            "file_path": f"{filed_path}/{filed_name}/top.mp4",
            "roi": [250, 1350, 780, 1880],
            "id": 1,
            "queue": "",
            "direction": "top"
        },
        "params_front": {
            "file_path": f"{filed_path}/{filed_name}/front.mp4",
            "roi": [450, 1440, 780, 1700],
            "id": 0,
            "direction": "front"
        },
        "params_rear": {
            "file_path": f"{filed_path}/{filed_name}/rear.mp4",
            "roi": [520, 1425, 480, 1790],
            "id": 1,
            "direction": "rear"
        }
    }
    return camera_config


def find_shortest_key(global_data):
    # 初始化最短键及其长度
    shortest_key = None
    shortest_length = float('inf')  # 设置初始长度为无穷大

    # 遍历字典中的每个键及其对应的列表
    for key, value in global_data.items():
        if len(value) < shortest_length:
            shortest_length = len(value)  # 更新最短长度
            shortest_key = key  # 更新最短键
    return shortest_key


def check_lists_equal_length(global_data):
    # 获取第一个键对应列表的长度
    lengths = [len(value) for value in global_data.values()]
    # 检查是否所有长度都相等
    return all(length == lengths[0] for length in lengths)


def check_and_remove(folder_path):
    # 需要判断的文件列表
    required_files = ["left.mp4", "right.mp4", "front.mp4", "rear.mp4", "top.mp4"]

    # 检查文件夹是否存在
    if os.path.exists(folder_path):
        # 检查必需文件是否存在
        all_files_exist = all(os.path.isfile(os.path.join(folder_path, file)) for file in required_files)

        if not all_files_exist:
            # 如果没有所有必需的文件，删除该文件夹及其所有内容
            shutil.rmtree(folder_path)
            print(f"文件夹 '{folder_path}' 及其所有内容已被删除。")
        else:
            print(f"文件夹 '{folder_path}' 中的文件存在，无需删除。")
    else:
        print(f"文件夹 '{folder_path}' 不存在。")


class RightFrontTopCaptureProcess(threading.Thread):
    def __init__(self, camera_info):
        super().__init__()
        self.file_path = camera_info['file_path']
        self.direction = camera_info['direction']
        self.roi = camera_info.get('roi', None)
        self.have_counter = 0
        self.not_counter = 0
        self.have_car_timer = 0
        self.not_car_timer = 0
        self.car_count = 0
        self.not_car_timer = [0, ]
        self.image_process_recognize()

    def image_process_recognize(self):
        self.trt_infer = trt_infer
        cfg_dict = {
            "engine_path": "./tianJinGang.engine",
            "class": [0, 1]  # 0 有车，result：1没车
        }
        self.mobilenet_wrapper = self.trt_infer.MobilenetTRT(cfg_dict["engine_path"])
        # 执行 warm-up 操作
        for i in range(10):
            thread1 = self.trt_infer.warmUpThread(self.mobilenet_wrapper)
            thread1.start()
            thread1.join()

    def run(self):
        global global_data
        count = 0
        try:
            cap = cv2.VideoCapture(self.file_path, cv2.CAP_FFMPEG, [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY])
            # 获取视频的帧率
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"帧率: {fps} 帧每秒")
            # 获取视频的总帧数
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            print(f"总帧数: {total_frames}")
            # 计算视频总时长（单位秒）
            total_duration = total_frames / fps
            print(f"视频总时长: {total_duration} 秒")
            while True:
                cap.grab()
                ret, frame = cap.retrieve()
                if not ret:
                    count += 1
                    if count >= 5:
                        break
                    continue
                # 获取当前视频的播放时间点（单位毫秒）
                current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # 转换为秒
                if current_time <= total_duration:
                    seg_frame = frame[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
                    thread1 = self.trt_infer.inferThread(self.mobilenet_wrapper, [seg_frame])
                    thread1.start()
                    thread1.join()
                    result = thread1.get_result()
                    if result == 0:
                        # 识别到有车
                        self.not_counter = 0
                        self.car_count += 1
                        self.have_counter += 1
                        if self.have_counter == 10:
                            if self.not_car_timer:
                                global_data[self.direction].append(min(self.not_car_timer))
                                print(min(self.not_car_timer), ":", self.direction)
                                self.not_car_timer = []
                    if result == 1:
                        # 识别到没车
                        self.have_counter = 0
                        self.not_counter += 1
                        if self.not_counter >= 3:
                            self.not_car_timer.append(round(float(current_time), 2))
            if self.car_count >= 1:
                if self.not_car_timer:
                    global_data[self.direction].append(max(self.not_car_timer))
            if self.car_count == 0:
                global_data[self.direction].append(0)
                global_data[self.direction].append(max(self.not_car_timer))
        except Exception as error:
            print(f"ImageCaptureProcess---{self.direction}:图片读取有问题:{error}")


class FrontRearCaptureProcess(threading.Thread):
    def __init__(self, camera_info):
        super().__init__()
        self.file_path = camera_info['file_path']
        self.direction = camera_info['direction']
        self.roi = camera_info.get('roi', None)
        self.have_counter = 0
        self.not_counter = 0
        self.have_car_timer = 0
        self.not_car_timer = [0, ]
        self.car_count = 0
        self.initialize_inference()

    def initialize_inference(self):
        PLUGIN_LIBRARY = "./myplugins.dll"
        engine_file_path = "truck.engine"
        ctypes.CDLL(PLUGIN_LIBRARY)
        self.csd_detector = cont_trt_infer.CSD_Detector(engine_file_path)  # 初始化detector
        self.my_container_detect = cont_trt_infer.container_detect(self.csd_detector)

    def run(self):
        global global_data
        count = 0
        print(self.direction)
        try:
            cap = cv2.VideoCapture(self.file_path, cv2.CAP_FFMPEG, [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY])
            # 获取视频的帧率
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"帧率: {fps} 帧每秒")
            # 获取视频的总帧数
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            print(f"总帧数: {total_frames}")
            # 计算视频总时长（单位秒）
            total_duration = total_frames / fps
            print(f"视频总时长: {total_duration} 秒")
            while True:
                cap.grab()
                ret, frame = cap.retrieve()
                if not ret:
                    count += 1
                    if count >= 5:
                        break
                    continue
                # 获取当前视频的播放时间点（单位毫秒）
                current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # 转换为秒
                seg_frame = frame[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
                self.my_container_detect.process_frame(seg_frame)
                _, reuslt_dict = self.my_container_detect.get_result()
                if self.my_container_detect.non_truck > 10 and self.my_container_detect.new_truck:
                    self.not_counter = 0
                    self.have_counter += 1
                    # if self.have_counter == 2:
                    self.car_count += 1
                    if self.not_car_timer:
                        if len(global_data[self.direction]) == 0:
                            global_data[self.direction].append(min(self.not_car_timer))
                            print(min(self.not_car_timer), ":", self.direction)
                        else:
                            global_data[self.direction].append(max(self.not_car_timer))
                            print(max(self.not_car_timer), ":", self.direction)
                    self.not_car_timer = []
                    self.my_container_detect.new_truck = False
                    self.my_container_detect.max_area_dict.clear()
                    self.my_container_detect.res_dict.clear()

                else:
                    # 识别到没车
                    self.have_counter = 0
                    self.not_counter += 1
                    # if self.not_counter >= 2:
                    self.not_car_timer.append(round(float(current_time), 2))

            if self.car_count >= 1:
                if self.not_car_timer:
                    global_data[self.direction].append(max(self.not_car_timer))
            if self.car_count == 0:
                global_data[self.direction].append(max(self.not_car_timer))
        except Exception as error:
            print(f"ImageCaptureProcess---{self.direction}:图片读取有问题:{error}")


# 截取视频的函数
def cut_video(input_video_path, output_video_path, start_time, end_time):
    now = datetime.now()
    # 视频文件列表
    vedio_list = ["left.mp4", "right.mp4", "front.mp4", "rear.mp4", "top.mp4"]
    folder_name = now.strftime("%Y%m%d%H%M%S")
    output_video_path = os.path.join(output_video_path, folder_name)
    os.makedirs(output_video_path, exist_ok=True)
    print(f"文件夹 '{folder_name}' 创建成功")
    for i in vedio_list:
        input_video_path_result = os.path.join(input_video_path, i)  # 原始视频路径
        output_video_path_result = os.path.join(output_video_path, i)  # 输出视频路径
        cap = cv2.VideoCapture(input_video_path_result)
        start_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        # 加载原始视频
        with VideoFileClip(input_video_path_result) as video:
            print(f"原视频帧率: {start_fps}")
            # 截取从指定时间段的片段
            video_clip = video.subclip(start_time, end_time)
            # 写入新视频文件，保持相同的帧率
            video_clip.write_videofile(output_video_path_result, codec='libx264', audio_codec='aac', fps=start_fps)
            cap = cv2.VideoCapture(input_video_path_result)
            end_fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            print(f"{i} 视频截取完成！帧率为:{end_fps} \n")
    return output_video_path


if __name__ == '__main__':

    time_quantum = []
    filed_path = r"D:/TianJinGangTest/"
    # filed_name = "202411111601"  # 两辆车
    # filed_name = "202411111538"
    filed_name = "202412031458"
    capture_video_path = r"D:\TianJinGangTest\capture_video"

    thread_list = []
    camera_config = get_camera_info(filed_path, filed_name)
    front_rear_capture_processes = [FrontRearCaptureProcess(camera_info) for camera_info in camera_config.values() if
                                    camera_info['direction'] in ['front', 'rear']]
    for process in front_rear_capture_processes:
        thread_list.append(process)
        process.start()
    right_front_top_capture_processes = [RightFrontTopCaptureProcess(camera_info) for camera_info in camera_config.values() if
                                         camera_info['direction'] in ['right', 'left', 'top']]
    # camera_info['direction'] in ['left']]
    for process in right_front_top_capture_processes:
        thread_list.append(process)
        process.start()
    for thread_data in thread_list:
        thread_data.join()

    print(global_data)
    print(filter_adjacent(global_data))
    capture_video_data = filter_adjacent(global_data)
    front_value = capture_video_data.get('front', [])
    rear_value = capture_video_data.get('rear', [])
    if len(front_value) < len(rear_value):
        time_quantum = front_value
    else:
        time_quantum = rear_value

    time_quantum_lenght = len(time_quantum)
    if time_quantum_lenght >= 2:
        for index, value in enumerate(time_quantum):
            if index <= len(time_quantum) - 2:
                start_time = time_quantum[index]
                if start_time > 0:
                    start_time = int(start_time - 8)
                end_time = time_quantum[index + 1]
                if index != len(time_quantum) - 1:
                    end_time = int(end_time + 3)
                print(f"开始时间：{start_time}，结束时间：{end_time}")

                # cut_video_prosses = Process(target=cut_video, args=(filed_path + filed_name, capture_video_path, start_time, end_time))
                # cut_video_prosses.start()
                # cut_video_prosses.join()
                output_video_path = cut_video(filed_path + filed_name, capture_video_path, start_time, end_time)
                print("完成一个片段的截取:", start_time, end_time)
                check_and_remove(output_video_path)
                time.sleep(10)
