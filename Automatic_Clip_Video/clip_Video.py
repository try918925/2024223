# -*- coding: utf-8 -*-
import json
import os
os.add_dll_directory(r"C:/opencv-4.9.0/build/install/x64/vc16/bin")
os.add_dll_directory(r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin")
import cv2
import threading
import test_infer as trt_infer
import ctypes
import Container_det_trt_yolov5 as cont_trt_infer
import time

# 设置当前目录为工作目录且与
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
execution_path = CURRENT_DIR.replace("\\", "/")
print("execution_path:", execution_path)
global_data = {
    "front": [],
    "rear": [],
    "left": [],
    "top": [],
    "right": [],
}


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
        self.not_car_timer = [0, ]
        self.image_process_recognize()

    def image_process_recognize(self):
        self.trt_infer = trt_infer
        cfg_dict = {
            "engine_path": execution_path + "/tianJinGang.engine",
            "class": [0, 1]  # 0 有车，result：1没车
        }
        self.mobilenet_wrapper = self.trt_infer.MobilenetTRT(cfg_dict["engine_path"])
        # 执行 warm-up 操作
        for i in range(10):
            thread1 = self.trt_infer.warmUpThread(self.mobilenet_wrapper)
            thread1.start()
            thread1.join()

    def run(self):
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
                        self.have_counter += 1
                        if self.have_counter == 3:
                            global_data[self.direction].append(min(self.not_car_timer))
                            print(min(self.not_car_timer))
                            self.not_car_timer = []

                    if result == 1:
                        # 识别到没车
                        self.have_counter = 0
                        self.not_counter += 1
                        if self.not_counter >= 3:
                            self.not_car_timer.append(round(float(current_time), 2))

            if self.not_car_timer:
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
        self.not_car_timer = 0
        self.not_car_timer = [0, ]
        self.initialize_inference()

    def initialize_inference(self):
        PLUGIN_LIBRARY = execution_path + "/myplugins.dll"
        engine_file_path = execution_path + "/truck.engine"
        ctypes.CDLL(PLUGIN_LIBRARY)
        self.csd_detector = cont_trt_infer.CSD_Detector(engine_file_path)  # 初始化detector
        self.my_container_detect = cont_trt_infer.container_detect(self.csd_detector)

    def run(self):
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
                # print(f"当前视频时间: {current_time} 秒")
                seg_frame = frame[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
                self.my_container_detect.process_frame(seg_frame)
                _, reuslt_dict = self.my_container_detect.get_result()
                if reuslt_dict:
                    # 识别到有车
                    if self.direction == 'front':
                        cv2.imwrite(f"./test_frame/{time.time()}.jpg", seg_frame)
                    self.not_counter = 0
                    self.have_counter += 1
                    if self.have_counter == 3:
                        if self.not_car_timer:
                            global_data[self.direction].append(min(self.not_car_timer))
                            print(min(self.not_car_timer))
                            self.not_car_timer = []
                else:
                    # 识别到没车
                    self.have_counter = 0
                    self.not_counter += 1
                    if self.not_counter >= 3:
                        self.not_car_timer.append(round(float(current_time), 2))
            if self.not_car_timer:
                global_data[self.direction].append(max(self.not_car_timer))

        except Exception as error:
            print(f"ImageCaptureProcess---{self.direction}:图片读取有问题:{error}")


if __name__ == '__main__':
    thread_list = []
    with open('camera_model.json', 'r') as f:
        camera_config = json.load(f)
    front_rear_capture_processes = [FrontRearCaptureProcess(camera_info) for camera_info in camera_config.values() if
                                    camera_info['direction'] in ['front', 'rear']]
    for process in front_rear_capture_processes:
        thread_list.append(process)
        process.start()

    # right_front_top_capture_processes = [RightFrontTopCaptureProcess(camera_info) for camera_info in camera_config.values() if
    #                                      camera_info['direction'] in ['right', 'left', 'top']]
    # for process in right_front_top_capture_processes:
    #     thread_list.append(process)
    #     process.start()

    for thread_data in thread_list:
        thread_data.join()

    print(global_data)
