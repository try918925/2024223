# -*- coding: utf-8 -*-
import json
import os, zmq, socket, sys
import threading

os.add_dll_directory(r"C:/opencv_build_data/opencv-4.9.0/build/install/x64/vc16/bin")
os.add_dll_directory(r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin")
import cv2
import ctypes
import numpy as np
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *
from uuu import Ui_MainWindow
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Queue, Process, freeze_support
import test_infer as trt_infer
import infer_det_rec as det_ocr
import infer_det_rec_car as det_ocr_car
import stichImg as ts
from algorithms.detector import YOLOv5Detector
import Container_det_trt_yolov5 as cont_trt_infer
import time
from datetime import datetime
from configs import config_5s_trt as my_config
import copy
from datetime import datetime
import psutil
import GPUtil

now = datetime.now()
xiangti = ''


def camera_model_data(filed_name):
    return {
        "params_left": {
            "file_path": f"C:/test_veodie/22fps/{filed_name}/left.mp4",
            "roi": [400, 1300, 50, 950],
            "id": 0,
            "queue": "",
            "direction": "left"
        },
        "params_right": {
            "file_path": f"C:/test_veodie/22fps/{filed_name}/right.mp4",
            "roi": [400, 1300, 1650, 2550],
            "id": 2,
            "queue": "",
            "direction": "right"
        },
        "params_top": {
            "file_path": f"C:/test_veodie/22fps/{filed_name}/top.mp4",
            "roi": [250, 1350, 780, 1880],
            "id": 1,
            "queue": "",
            "direction": "top"
        },
        "params_front": {
            "file_path": f"C:/test_veodie/22fps/{filed_name}/front.mp4",
            "roi": [450, 1440, 780, 1700],
            "id": 0,
            "direction": "front"
        },
        "params_rear": {
            "file_path": f"C:/test_veodie/22fps/{filed_name}/rear.mp4",
            "roi": [520, 1425, 480, 1790],
            "id": 1,
            "direction": "rear"
        }
    }


class RightLeftCaptureProcess(Process):
    def __init__(self, camera_info, queue, sleep_time=0.02):
        super().__init__()
        self.camera_id = camera_info['id']
        self.file_path = camera_info['file_path']
        self.direction = camera_info['direction']
        self.roi = camera_info.get('roi', None)
        self.queue = queue
        self.front_rear = ("front", "rear")
        self.left_right = ("right", "left")
        self.dsize = (900, 600)
        self.roi_cols = (self.roi[2], self.roi[3])
        self.roi_rows = (self.roi[0], self.roi[1])
        self.sleep_time = sleep_time
        self.front_rear_count = 0
        self.counter = 0

    def run(self):
        # count = 0
        # start_time = time.time()
        try:
            decoder = cv2.cudacodec.createVideoReader(self.file_path)
            while True:
                time.sleep(self.sleep_time)
                ret, frame = decoder.nextFrame()
                if not ret or frame is None:
                    self.counter += 1
                    if self.counter >= 5:
                        decoder = cv2.cudacodec.createVideoReader(self.file_path)
                        print(f"{self.direction}:识别断流........")
                        self.counter = 0
                    continue

                if self.direction in self.front_rear:
                    frame_crop = frame.colRange(self.roi_cols[0], self.roi_cols[1]).rowRange(self.roi_rows[0],
                                                                                             self.roi_rows[1])
                    frame_cpu = frame_crop.download()  # 裁剪后的图像下载到 CPU
                    frame_cpu = frame_cpu[:, :, :3]  # 裁剪通道
                    self.queue.put((self.camera_id, frame_cpu))
                    self.front_rear_count = 0
                else:
                    frame_resized = cv2.cuda.resize(frame, self.dsize)  # 对原图进行 resize
                    frame_crop = frame.colRange(self.roi_cols[0], self.roi_cols[1]).rowRange(self.roi_rows[0],
                                                                                             self.roi_rows[1])
                    frame_cpu_crop = frame_crop.download()  # 裁剪后的图像
                    frame_cpu_resized = frame_resized.download()  # 调整大小后的图像
                    frame_cpu_crop = frame_cpu_crop[:, :, :3]
                    frame_cpu_resized = frame_cpu_resized[:, :, :3]
                    self.queue.put((self.camera_id, frame_cpu_resized, frame_cpu_crop))

                self.counter = 0

                # end_time = time.time() - start_time
                # count += 1
                # if end_time >= 120:
                #     print(f"{self.direction}：数量为:{count} 时间:{end_time} \n")
                #     count = 0
                #     start_time = time.time()

        except Exception as error:
            print(f"ImageCaptureProcess---{self.direction}:图片读取有问题:{error}")


class FrontRearCaptureProcess(threading.Thread):
    def __init__(self, camera_info, queue, sleep_time=0.02):
        super().__init__()
        self.camera_id = camera_info['id']
        self.file_path = camera_info['file_path']
        self.direction = camera_info['direction']
        self.roi = camera_info.get('roi', None)
        self.queue = queue
        self.front_rear = ("front", "rear")
        self.left_right = ("right", "left")
        self.dsize = (900, 600)
        self.roi_cols = (self.roi[2], self.roi[3])
        self.roi_rows = (self.roi[0], self.roi[1])
        self.sleep_time = sleep_time
        self.front_rear_count = 0
        self.counter = 0

    def run(self):
        # count = 0
        # start_time = time.time()
        try:
            decoder = cv2.cudacodec.createVideoReader(self.file_path)
            while True:
                time.sleep(self.sleep_time)
                ret, frame = decoder.nextFrame()
                if not ret or frame is None:
                    self.counter += 1
                    if self.counter >= 5:
                        decoder = cv2.cudacodec.createVideoReader(self.file_path)
                        print(f"{self.direction}:识别断流........")
                        self.counter = 0
                    continue

                if self.direction in self.front_rear:
                    frame_crop = frame.colRange(self.roi_cols[0], self.roi_cols[1]).rowRange(self.roi_rows[0],
                                                                                             self.roi_rows[1])
                    frame_cpu = frame_crop.download()  # 裁剪后的图像下载到 CPU
                    frame_cpu = frame_cpu[:, :, :3]  # 裁剪通道
                    self.queue.put((self.camera_id, frame_cpu))
                    self.front_rear_count = 0
                else:
                    frame_resized = cv2.cuda.resize(frame, self.dsize)  # 对原图进行 resize
                    frame_crop = frame.colRange(self.roi_cols[0], self.roi_cols[1]).rowRange(self.roi_rows[0],
                                                                                             self.roi_rows[1])
                    frame_cpu_crop = frame_crop.download()  # 裁剪后的图像
                    frame_cpu_resized = frame_resized.download()  # 调整大小后的图像
                    frame_cpu_crop = frame_cpu_crop[:, :, :3]
                    frame_cpu_resized = frame_cpu_resized[:, :, :3]
                    self.queue.put((self.camera_id, frame_cpu_resized, frame_cpu_crop))

                self.counter = 0

                # end_time = time.time() - start_time
                # count += 1
                # if end_time >= 120:
                #     print(f"{self.direction}：数量为:{count} 时间:{end_time} \n")
                #     count = 0
                #     start_time = time.time()

        except Exception as error:
            print(f"ImageCaptureProcess---{self.direction}:图片读取有问题:{error}")


class TopCaptureProcess(threading.Thread):
    def __init__(self, camera_info, queue, sleep_time=0.02):
        super().__init__()
        self.camera_id = camera_info['id']
        self.file_path = camera_info['file_path']
        self.direction = camera_info['direction']
        self.roi = camera_info.get('roi', None)
        self.queue = queue
        self.front_rear = ("front", "rear")
        self.left_right = ("right", "left")
        self.dsize = (900, 600)
        self.roi_cols = (self.roi[2], self.roi[3])
        self.roi_rows = (self.roi[0], self.roi[1])
        self.sleep_time = sleep_time
        self.front_rear_count = 0
        self.counter = 0

    def run(self):
        # count = 0
        # start_time = time.time()
        try:
            decoder = cv2.cudacodec.createVideoReader(self.file_path)
            while True:
                time.sleep(self.sleep_time)
                ret, frame = decoder.nextFrame()
                if not ret or frame is None:
                    self.counter += 1
                    if self.counter >= 5:
                        decoder = cv2.cudacodec.createVideoReader(self.file_path)
                        print(f"{self.direction}:识别断流........")
                        self.counter = 0
                    continue

                if self.direction in self.front_rear:
                    frame_crop = frame.colRange(self.roi_cols[0], self.roi_cols[1]).rowRange(self.roi_rows[0],
                                                                                             self.roi_rows[1])
                    frame_cpu = frame_crop.download()  # 裁剪后的图像下载到 CPU
                    frame_cpu = frame_cpu[:, :, :3]  # 裁剪通道
                    self.queue.put((self.camera_id, frame_cpu))
                    self.front_rear_count = 0
                else:
                    frame_resized = cv2.cuda.resize(frame, self.dsize)  # 对原图进行 resize
                    frame_crop = frame.colRange(self.roi_cols[0], self.roi_cols[1]).rowRange(self.roi_rows[0],
                                                                                             self.roi_rows[1])
                    frame_cpu_crop = frame_crop.download()  # 裁剪后的图像
                    frame_cpu_resized = frame_resized.download()  # 调整大小后的图像
                    frame_cpu_crop = frame_cpu_crop[:, :, :3]
                    frame_cpu_resized = frame_cpu_resized[:, :, :3]
                    self.queue.put((self.camera_id, frame_cpu_resized, frame_cpu_crop))

                self.counter = 0

                # end_time = time.time() - start_time
                # count += 1
                # if end_time >= 120:
                #     print(f"{self.direction}：数量为:{count} 时间:{end_time} \n")
                #     count = 0
                #     start_time = time.time()

        except Exception as error:
            print(f"ImageCaptureProcess---{self.direction}:图片读取有问题:{error}")


class RightProcessWorker(QThread):
    def __init__(self, camera_info, result_queue, ocr_queue, resource_queue):
        super().__init__()
        self.camera_id = camera_info['id']
        self.file_path = camera_info['file_path']
        self.direction = camera_info['direction']
        self.roi = camera_info.get('roi', None)
        self.result_queue = result_queue
        self.ocr_queue = ocr_queue
        self.resource_queue = resource_queue
        self.true_threshold = 3
        self.detector = None
        self.detector = YOLOv5Detector.from_config(my_config)
        self.my_stiching = ts.stiching_distribution()
        self.executor = ThreadPoolExecutor(max_workers=5)

    def call_shuangxiang(self, frame):
        global xiangti
        img = np.ascontiguousarray(frame)
        # 保证数据和训练数据尺寸统一
        img = cv2.resize(img, (4160, 1040))
        obj_list = self.detector.det([img])[0]
        if len(obj_list) == 2:
            self.resource_queue.put(('recognition', "Yl", ['双箱']))
            xiangti = '双箱'
        else:
            self.resource_queue.put(("recognition", "Yl", ['单箱']))
            xiangti = '单箱'

    def process_frames(self, frames, id, num, status):
        # start_time = time.time()
        if id == 2 and self.direction == "right":
            result, state = self.my_stiching.stiching(frames, "right", num, status)
        if state:
            if id == 2:
                self.executor.submit(self.call_shuangxiang, result)
            resized_image = cv2.resize(result, (1278, 344))
            self.resource_queue.put(("frame", self.direction, resized_image))
            cv2.imwrite(f'tmp/{self.direction}.jpg', resized_image)

    def run(self):
        self.frames_to_process = []
        self.car_in = False
        self.consecutive_true_count = 0
        self.consecutive_zero_count = 0
        self.count = 0
        self.count_all = 0
        while True:
            if not self.result_queue.empty():
                self.camera_id, result, frame = self.result_queue.get()
                # print(f"{self.direction}:ImageProcessWorker:队列长度为:{self.result_queue.qsize()}")
                if result == 1:
                    self.consecutive_zero_count = 0
                    self.consecutive_true_count += 1
                    if self.consecutive_true_count >= self.true_threshold:
                        # 连续无车
                        self.consecutive_true_count = 0
                        if len(self.frames_to_process) < 5:
                            # print(f"清空列表1:self.frames_to_process", result)
                            self.frames_to_process = []
                        # 标志位，代表车辆是否有进入记录
                        if self.car_in:
                            self.car_in = False
                            self.resource_queue.put(("lane", self.direction, 0))
                            self.count += 1
                            self.count_all += len(self.frames_to_process)
                            print(self.direction, "传入的总图数量为：", self.count_all)

                            self.executor.submit(self.process_frames, self.frames_to_process, self.camera_id, self.count, True)
                            self.count = 0
                            self.count_all = 0
                            self.frames_to_process = []

                if result == 0:
                    self.consecutive_true_count = 0
                    self.consecutive_zero_count += 1
                    self.frames_to_process.append(frame)
                    if len(self.frames_to_process) >= 50:
                        self.count += 1
                        self.count_all += len(self.frames_to_process)

                        self.executor.submit(self.process_frames, self.frames_to_process, self.camera_id, self.count, False)
                        self.frames_to_process = []
                    if self.consecutive_zero_count > 3:
                        self.car_in = True
                        self.resource_queue.put(("lane", self.direction, 1))
                if result == "NO" and self.car_in:
                    self.frames_to_process.append(frame)


class LeftProcessWorker(threading.Thread):
    def __init__(self, camera_info, result_queue, ocr_queue, resource_queue):
        super().__init__()
        self.camera_id = camera_info['id']
        self.file_path = camera_info['file_path']
        self.direction = camera_info['direction']
        self.roi = camera_info.get('roi', None)
        self.result_queue = result_queue
        self.ocr_queue = ocr_queue
        self.resource_queue = resource_queue
        self.true_threshold = 3
        self.detector = None
        self.my_stiching = None
        self.executor = None

    def process_frames(self, frames, id, num, status):
        if id == 0:
            result, state = self.my_stiching.stiching(frames, "left", num, status)
        if state:
            resized_image = cv2.resize(result, (1278, 344))
            self.resource_queue.put(("frame", self.direction, resized_image))
            cv2.imwrite(f'tmp/{self.direction}.jpg', resized_image)

    def run(self):
        self.my_stiching = ts.stiching_distribution()
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.frames_to_process = []
        self.car_in = False
        self.consecutive_true_count = 0
        self.consecutive_zero_count = 0
        self.count = 0
        self.count_all = 0
        while True:
            if not self.result_queue.empty():
                self.camera_id, result, frame = self.result_queue.get()
                # print(f"{self.direction}:ImageProcessWorker:队列长度为:{self.result_queue.qsize()}")
                if result == 1:
                    self.consecutive_zero_count = 0
                    self.consecutive_true_count += 1
                    if self.consecutive_true_count >= self.true_threshold:
                        # 连续无车
                        self.consecutive_true_count = 0
                        if len(self.frames_to_process) < 5:
                            # print(f"清空列表1:self.frames_to_process", result)
                            self.frames_to_process = []
                        # 标志位，代表车辆是否有进入记录
                        if self.car_in:
                            self.car_in = False
                            self.resource_queue.put(("lane", self.direction, 0))
                            self.count += 1
                            self.count_all += len(self.frames_to_process)
                            print(self.direction, "传入的总图数量为：", self.count_all)

                            self.executor.submit(self.process_frames, self.frames_to_process, self.camera_id, self.count, True)
                            self.count = 0
                            self.count_all = 0
                            self.frames_to_process = []

                if result == 0:
                    self.consecutive_true_count = 0
                    self.consecutive_zero_count += 1
                    self.frames_to_process.append(frame)
                    if len(self.frames_to_process) >= 50:
                        self.count += 1
                        self.count_all += len(self.frames_to_process)

                        self.executor.submit(self.process_frames, self.frames_to_process, self.camera_id, self.count, False)
                        self.frames_to_process = []
                    if self.consecutive_zero_count > 3:
                        self.car_in = True
                        self.resource_queue.put(("lane", self.direction, 1))
                if result == "NO" and self.car_in:
                    self.frames_to_process.append(frame)


class TopProcessWorker(threading.Thread):
    def __init__(self, camera_info, result_queue, ocr_queue, resource_queue):
        super().__init__()
        self.camera_id = camera_info['id']
        self.file_path = camera_info['file_path']
        self.direction = camera_info['direction']
        self.roi = camera_info.get('roi', None)
        self.result_queue = result_queue
        self.ocr_queue = ocr_queue
        self.resource_queue = resource_queue
        self.true_threshold = 3
        self.detector = None
        self.my_stiching = None
        self.executor = None

    def process_frames(self, frames, id, num, status):
        if id == 1:
            result, state = self.my_stiching.stiching(frames, "top", num, status)
        if state:
            resized_image = cv2.resize(result, (1278, 344))
            self.resource_queue.put(("frame", self.direction, resized_image))
            cv2.imwrite(f'tmp/{self.direction}.jpg', resized_image)

    def run(self):
        self.my_stiching = ts.stiching_distribution()
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.frames_to_process = []
        self.car_in = False
        self.consecutive_true_count = 0
        self.consecutive_zero_count = 0
        self.count = 0
        self.count_all = 0
        while True:
            if not self.result_queue.empty():
                self.camera_id, result, frame = self.result_queue.get()
                # print(f"{self.direction}:ImageProcessWorker:队列长度为:{self.result_queue.qsize()}")
                if result == 1:
                    self.consecutive_zero_count = 0
                    self.consecutive_true_count += 1
                    if self.consecutive_true_count >= self.true_threshold:
                        # 连续无车
                        self.consecutive_true_count = 0
                        if len(self.frames_to_process) < 5:
                            # print(f"清空列表1:self.frames_to_process", result)
                            self.frames_to_process = []
                        # 标志位，代表车辆是否有进入记录
                        if self.car_in:
                            self.car_in = False
                            self.resource_queue.put(("lane", self.direction, 0))
                            self.count += 1
                            self.count_all += len(self.frames_to_process)
                            print(self.direction, "传入的总图数量为：", self.count_all)

                            self.executor.submit(self.process_frames, self.frames_to_process, self.camera_id, self.count, True)
                            self.count = 0
                            self.count_all = 0
                            self.frames_to_process = []

                if result == 0:
                    self.consecutive_true_count = 0
                    self.consecutive_zero_count += 1
                    self.frames_to_process.append(frame)
                    if len(self.frames_to_process) >= 50:
                        self.count += 1
                        self.count_all += len(self.frames_to_process)

                        self.executor.submit(self.process_frames, self.frames_to_process, self.camera_id, self.count, False)
                        self.frames_to_process = []
                    if self.consecutive_zero_count > 3:
                        self.car_in = True
                        self.resource_queue.put(("lane", self.direction, 1))
                if result == "NO" and self.car_in:
                    self.frames_to_process.append(frame)


class ImageProcessRecognize(threading.Thread):
    def __init__(self, original_queue, result_queue, log_queue):
        super().__init__()
        self.original_queue = original_queue
        self.result_queue = result_queue
        self.log_queue = log_queue
        self.count = 0
        self.count_top = 0
        self.direction = {
            0: "left",
            1: "top",
            2: "right"
        }
        self.mean_data = []

    def run(self):
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
        while True:
            if not self.original_queue.empty():
                camera_id, frame, seg_frame = self.original_queue.get()
                self.count += 1
                self.count_top += 1
                if self.count % 2 == 0 and camera_id != 1:
                    start_time = time.time()
                    thread1 = self.trt_infer.inferThread(self.mobilenet_wrapper, [seg_frame])
                    thread1.start()
                    thread1.join()
                    result = thread1.get_result()
                    # print(f"ImageProcessRecognize识别时间:{time.time() - start_time}")
                    self.mean_data.append(time.time() - start_time)
                    if result == 0:
                        frame = cv2.resize(frame, (2560, 1440))
                    self.result_queue.put((camera_id, result, frame))
                    self.count = 0
                else:
                    if camera_id != 1:
                        frame = cv2.resize(frame, (2560, 1440))
                        self.result_queue.put((camera_id, "NO", frame))
                if camera_id == 1:
                    start_time = time.time()
                    thread1 = self.trt_infer.inferThread(self.mobilenet_wrapper, [seg_frame])
                    thread1.start()
                    thread1.join()
                    result = thread1.get_result()
                    self.mean_data.append(time.time() - start_time)
                    if result == 0:
                        frame = cv2.resize(frame, (2560, 1440))
                    self.result_queue.put((camera_id, result, frame))
                    self.count_top = 0

                if len(self.mean_data) >= 200:
                    print(f"{self.direction[camera_id]}:ImageProcessRecognize:队列长度为:{self.original_queue.qsize()}")
                    self.log_queue.put(f"{self.direction[camera_id]}:ImageProcessRecognize:队列长度为:{self.original_queue.qsize()}")
                    print(f"{self.direction[camera_id]}:ImageProcessRecognize识别无车时间:{np.mean(self.mean_data)}")
                    self.log_queue.put(f"{self.direction[camera_id]}:ImageProcessRecognize识别无车时间:{np.mean(self.mean_data)}")
                    self.mean_data.clear()


class ImageProcessWorker2(threading.Thread):
    def __init__(self, camera_info, qu1, qu2, qu3, resource_queue, log_queue):
        super().__init__()
        self.camera_id = camera_info['id']
        self.file_path = camera_info['file_path']
        self.direction = camera_info['direction']
        self.roi = camera_info.get('roi', None)
        self.queue = qu1
        self.rec_queue = qu2
        self.ocr_queue = qu3
        self.ocr_queue = qu3
        self.resource_queue = resource_queue
        self.log_queue = log_queue
        self.frame_counter = 0
        self.mean_data = []

    def initialize_inference(self):
        PLUGIN_LIBRARY = "./myplugins.dll"
        engine_file_path = "truck.engine"
        ctypes.CDLL(PLUGIN_LIBRARY)
        self.csd_detector = cont_trt_infer.CSD_Detector(engine_file_path)  # 初始化detector
        self.my_container_detect = cont_trt_infer.container_detect(self.csd_detector)

    def delete_old_files(self):
        try:
            # 获取tmp文件夹下的所有文件
            files = os.listdir('./tmp')
            # 如果文件数量大于1，则表示有旧文件需要删除
            for file in files:
                file_path = os.path.join('./tmp', file)
                os.remove(file_path)
            print("Old files deleted successfully.")
        except Exception as e:
            print("Error occurred while deleting old files:", e)

    def run(self):
        self.initialize_inference()
        res_dict_lst = []

        while True:
            if not self.queue.empty():
                self.camera_id, frame = self.queue.get()
                self.frame_counter += 1
                # rec队列采集 front==0
                if self.camera_id == 0:
                    if self.frame_counter == 4:
                        self.rec_queue.put((self.camera_id, frame))
                        self.frame_counter = 0
                start_time = time.time()
                self.my_container_detect.process_frame(frame)
                _, reuslt_dict = self.my_container_detect.get_result()
                if reuslt_dict:
                    res_dict_lst.append(reuslt_dict)
                # print(f"ImageProcessWorker2识别时间:{time.time() - start_time}")
                self.mean_data.append(time.time() - start_time)

                if self.my_container_detect.non_truck > 8 and self.my_container_detect.new_truck:
                    print(f"{self.direction}:ImageProcessWorker2:队列长度为:{self.queue.qsize()}")
                    self.log_queue.put(f"{self.direction}:ImageProcessWorker2:队列长度为:{self.queue.qsize()}")
                    print(f"{self.direction}:ImageProcessWorker2有车来:{time.time() - start_time}")
                    self.log_queue.put(f"{self.direction}:ImageProcessWorker2有车来:{time.time() - start_time}")
                    # 当连续12帧(约1s)没有集装箱面，且之前有卡车进入时，获取前一段时间面积最大帧
                    reuslt_dict, _ = self.my_container_detect.get_result()
                    self.clear_writer_img(self.camera_id, reuslt_dict['img'])
                    # ocr队列采集
                    self.ocr_queue.put(((self.camera_id + 6), reuslt_dict['img']))
                    final_label, _ = self.my_container_detect.door_label_vote(res_dict_lst)
                    # 界面刷新
                    self.resource_queue.put(("frame", self.direction, reuslt_dict['img']))
                    if final_label == 'door':
                        self.ocr_queue.put(((self.camera_id + 10), None))
                    if self.camera_id == 1:
                        self.rec_queue.put((5, None))
                        self.ocr_queue.put((5, None))
                    self.my_container_detect.new_truck = False
                    self.my_container_detect.max_area_dict.clear()
                    self.my_container_detect.res_dict.clear()
                    res_dict_lst.clear()

                if len(self.mean_data) >= 100:
                    print(f"{self.direction}:ImageProcessWorker2识别无车时间:{np.mean(self.mean_data)}")
                    self.log_queue.put(f"{self.direction}:ImageProcessWorker2识别无车时间:{np.mean(self.mean_data)}")
                    self.mean_data.clear()

    def clear_writer_img(self, camera_id, frame):
        file_path = f'tmp/{self.direction}.jpg'
        if camera_id == 0:
            self.delete_old_files()
        cv2.imwrite(file_path, frame)


class OcrThread(threading.Thread):
    def __init__(self, queue, resource_queue, log_queue):
        super().__init__()
        self.queue = queue
        self.log_queue = log_queue
        self.resource_queue = resource_queue
        self.mean_data = []

    def write_frame(self, path, frame):
        cv2.imwrite(path, frame)

    def run(self):
        self.config_dict = {
            "ocr_det_config": "./config/det/my_det_r50_db++_td_tr.yml",
            "ocr_rec_config": "./config/rec/my_en_PP-OCRv3_rec.yml"
        }
        self.ocr = det_ocr.OCR_process(self.config_dict)
        ocr_data_dict = {"front": "", "rear": ""}
        result_list = []
        result = None
        while True:
            id, frame = self.queue.get()
            if id == 10:
                ocr_data_dict["front"] = "箱门朝前"
                self.resource_queue.put(("recognition", "Ocr", ["箱门朝前"]))
            if id == 11:
                ocr_data_dict["rear"] = "箱门朝后"
                self.resource_queue.put(("recognition", "Ocr", ["箱门朝后"]))
            if id != 5:
                start_time = time.time()
                if id == 0:
                    result = self.ocr.process_imgs([frame], "left")
                if id == 2:
                    result = self.ocr.process_imgs([frame], "right")
                if id == 6:
                    frame_ues = cv2.GaussianBlur(frame, (5, 5), 0)
                    result = self.ocr.process_imgs([frame_ues, frame], "front")
                if id == 7:
                    frame_ues = cv2.GaussianBlur(frame, (5, 5), 0)
                    result = self.ocr.process_imgs([frame_ues, frame], "rear")

                if result is not None:
                    result1, result2 = result
                    if (result1 != '') & (result2 != ''):
                        result_list.append(result)
                    elif result1 != '' and result2 == '':
                        result_list.append((result1, ''))
                    elif result1 == '' and result2 != '':
                        result_list.append(('', result2))
                self.mean_data.append(time.time() - start_time)

            if len(self.mean_data) >= 2:
                print(f"OcrThread识别无车时间:{np.mean(self.mean_data)}")
                self.log_queue.put(f"OcrThread识别无车时间:{np.mean(self.mean_data)}")
                self.mean_data.clear()

            if id == 5:
                ultra_result_one = det_ocr.vote2res(result_list)
                result_list = []
                self.resource_queue.put(("recognition", "Ocr", ultra_result_one))
                self.write_ocr_data(ultra_result_one, copy.deepcopy(ocr_data_dict))
                ocr_data_dict = {"front": "", "rear": ""}
                print(f"OcrThread:队列长度为:{self.queue.qsize()}")
                self.log_queue.put(f"OcrThread:队列长度为:{self.queue.qsize()}")
                print(f"OcrThread有车识别时间:{time.time() - start_time}")
                self.log_queue.put(f"OcrThread有车识别时间:{time.time() - start_time}")

            time.sleep(0.001)

    def write_ocr_data(self, ultra_result, ocr_data_dict):
        file_path = 'tmp/ocr.txt'
        # 写入结果到文件中
        with open(file_path, 'w', encoding='utf-8') as file:
            ultra_result = ultra_result[:-4] + ' ' + ultra_result[-4:]
            self.log_queue.put(f"箱号:{ultra_result}")
            file.write(ultra_result + '\n')
            if ocr_data_dict['front']:
                file.write(ocr_data_dict['front'] + '\n')
            if ocr_data_dict['rear']:
                file.write(ocr_data_dict['rear'] + '\n')


class RecThread(threading.Thread):
    def __init__(self, queue, resource_queue, log_queue):
        super().__init__()
        self.queue = queue
        self.resource_queue = resource_queue
        self.log_queue = log_queue
        self.cont_num = 1
        self.mean_data = []
        self.lp = None

    def run(self):
        self.weights_dict = {
            "ocr_det_config": "./config_car/det/my_car_det_r50_db++_td_tr.yml",
            "ocr_rec_config": "./config_car/rec/my_rec_chinese_lite_train_v2.0.yml"
        }
        self.lp = det_ocr_car.OCR_process(self.weights_dict)
        result_list = []
        save_frame = []

        while True:
            id, frame = self.queue.get()
            if id != 5:
                start_time = time.time()
                use_frame = cv2.GaussianBlur(frame, (5, 5), 0)
                result = self.lp.process_imgs([use_frame])
                if len(result) > 0:
                    result_list.append([result, frame])
                    save_frame.append(frame)
                # print(f"RecThread识别时间:{time.time() - start_time}")
                self.mean_data.append(time.time() - start_time)

            if len(self.mean_data) >= 20:
                print(f"RecThread无车识别时间:{np.mean(self.mean_data)}")
                self.log_queue.put(f"RecThread无车识别时间:{np.mean(self.mean_data)}")
                self.mean_data.clear()

            # print(f"RecThread:use_frame  时间:{time.time() - start_time} \n")

            if id == 5:
                start_time = time.time()
                ultra_result, save_frame_idx = det_ocr_car.get_finalResult(result_list)
                if len(save_frame) > 0:
                    final_save = save_frame[save_frame_idx]
                    self.writer_img('tmp/chepai.jpg', final_save)
                self.recMessage.emit(1, ultra_result)
                # self.record_reccar(ultra_result)
                print(f"RecThread:队列长度为:{self.queue.qsize()}")
                self.log_queue.put(f"RecThread:队列长度为:{self.queue.qsize()}")
                print(f"RecThread有车识别时间:{time.time() - start_time}")
                self.log_queue.put(f"RecThread有车识别时间:{time.time() - start_time}")
                result_list = []
                save_frame = []
                self.writer_rec_data(ultra_result)
                self.log_queue.put(f"RecThread号:{ultra_result}")
            time.sleep(0.0001)

    def record_reccar(self, ultra_result):
        with open("0611-001-reccar.txt", "a") as fw:
            fw.write(str(self.cont_num) + "/t" + ultra_result + "\n")
            self.cont_num += 1

    def writer_rec_data(self, ultra_result):
        file_path = 'tmp/rec.txt'
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(ultra_result)

    def writer_img(self, file_path, frame):
        cv2.imwrite(file_path, frame)


class ComputationalResource(QThread):
    recocrylMessage = Signal(str, list)
    frameData = Signal(str, object)
    laneData = Signal(str, int)

    def __init__(self, resource_queue):
        super().__init__()
        self.queue = resource_queue

    def run(self):
        while True:
            if not self.queue.empty():
                account, place, frame = self.queue.get()
                if account == "frame":
                    self.frameData.emit(place, frame)
                elif account == "recognition":
                    self.recocrylMessage.emit(place, frame)
                # account
                else:
                    self.laneData.emit(place, frame)
            time.sleep(0.02)


class SaveProcessWorker(QThread):
    def __init__(self, path, channel, log_queue):
        super().__init__()
        self.path = path
        self.ch = channel
        self.image_number = '00'
        self.vehicle_count = "CAR0001"
        self.ip_address = socket.gethostbyname(socket.gethostname())
        # ZeroMQ 上下文
        self.context = zmq.Context()
        # 创建发布者Socket，并绑定到指定端口
        self.publisher = self.context.socket(zmq.PUB)
        self.publisher.bind("tcp://*:5555")  # 本地绑定，使用端口5555
        self.files_to_check = ['top.jpg', 'left.jpg', 'right.jpg', 'rear.jpg', 'front.jpg', 'ocr.txt', 'rec.txt']
        self.not_send_files = ['ocr.txt', 'rec.txt']
        self.log_queue = log_queue

    def rename_file(self, file_path, output_path):
        max_retries = 3
        retries = 0
        while retries < max_retries:
            try:
                os.rename(file_path, output_path)
                break  # 如果成功，退出循环
            except PermissionError:
                retries += 1
                time.sleep(1)  # 等待一段时间后重试
        else:
            print("无法重命名文件，达到最大重试次数。")

    def check(self):
        current_date = datetime.now().strftime("%y%m%d")
        check_floder = os.path.join(self.path, self.ch, current_date)
        if os.path.exists(check_floder):
            files = os.listdir(check_floder)
            if files:
                sorted_files = sorted(files, reverse=True)
                vehicle_count = sorted_files[0]
                prefix = vehicle_count[:-4]
                suffix = vehicle_count[-4:]
                new_suffix = str(int(suffix) + 1).zfill(4)
                vehicle_count = f"{prefix}{new_suffix}"
                self.vehicle_count = vehicle_count

        output_folder = os.path.join(check_floder, self.vehicle_count)
        os.makedirs(output_folder, exist_ok=True)
        self.check_and_save(output_folder)

    def check_and_save(self, output_folder):
        li = []
        for tmp_data in self.files_to_check:
            file_path = f'tmp/{tmp_data}'
            output_path = os.path.join(output_folder, tmp_data)
            self.rename_file(file_path, output_path)
            if tmp_data not in self.not_send_files:
                li.append(output_path)

        if os.path.exists('tmp/chepai.jpg'):
            self.rename_file("tmp/chepai.jpg", os.path.join(output_folder, "chepai.jpg"))

        try:
            global xiangti
            if xiangti == "":
                time.sleep(3)
                self.save_xiangti(output_folder, xiangti)
            xiangti = ''
        except Exception as e:
            # 在这里处理异常
            print("An error occurred:", e, "\n")

        self.send_zmq_string(li)

    def save_xiangti(self, output_folder, xiangti):
        ocr_path = os.path.join(output_folder, "ocr.txt")
        with open(ocr_path, 'a', encoding='utf-8') as file:
            file.write(xiangti + '\n')

    def send_zmq_string(self, li):
        # 获取当前时间
        current_time = datetime.now()
        # 格式化时间戳
        timestamp = current_time.strftime("%Y%m%d%H%M%S")
        # 构建数据字符串
        data = f"[C|{timestamp}|{self.ch}|{self.vehicle_count}|{li[3]}|{li[4]}|{li[0]}|{li[2]}|{li[1]}|{self.ip_address}|D]"
        # 发布数据
        self.publisher.send_string(data)

    def check_files_existence(self, folder_path):
        # 检查每个文件是否存在
        for file_name in self.files_to_check:
            file_path = os.path.join(folder_path, file_name)
            if not os.path.exists(file_path):
                return False
        return True

    def run(self):
        folder_path = 'tmp'
        while True:
            start_time = time.time()
            if self.check_files_existence(folder_path):
                self.check()
                print("保存数据时间:", time.time() - start_time)
                self.log_queue.put(f"保存数据时间:{time.time() - start_time}")
            time.sleep(2)


def rear_front_rec_ocr_one_process(camera_info, rec_queue, ocr_queue, resource_queue, log_queue):
    thread_list = []
    workers = [ImageProcessWorker2(camera_info, queue, rec_queue, ocr_queue, resource_queue, log_queue) for camera_info, queue in camera_info]
    for worker in workers:
        thread_list.append(worker)
        worker.start()
    rec_thread = RecThread(rec_queue, resource_queue, log_queue)
    thread_list.append(rec_thread)
    rec_thread.start()

    ocr_thread = OcrThread(ocr_queue, resource_queue, log_queue)
    thread_list.append(ocr_thread)
    ocr_thread.start()

    for thread_data in thread_list:
        thread_data.join()


def left_right_top_recognize_process(original_result_queue, log_queue):
    thread_list = []
    recognizes = [ImageProcessRecognize(original_queues, result_queues, log_queue) for original_queues, result_queues in original_result_queue]
    for recognize in recognizes:
        thread_list.append(recognize)
        recognize.start()

    for thread_data in thread_list:
        thread_data.join()


def front_rear_capture_process(camera_info):
    thread_list = []
    for camera, original_queues in camera_info:
        if camera["direction"] in ["front", "rear"]:
            process = FrontRearCaptureProcess(camera, original_queues, sleep_time=0.065)
            thread_list.append(process)
            process.start()
        if camera["direction"] in ["top"]:
            process = TopCaptureProcess(camera, original_queues, sleep_time=0.065)
            thread_list.append(process)
            process.start()

    for thread_data in thread_list:
        thread_data.join()


def top_left_worker_process(camera_info, ocr_queue, resource_queue):
    thread_list = []
    for camera, result_queues in camera_info:
        if camera["direction"] in ["top"]:
            process = TopProcessWorker(camera, result_queues, ocr_queue, resource_queue)
            thread_list.append(process)
            process.start()

        if camera["direction"] in ["left"]:
            process = LeftProcessWorker(camera, result_queues, ocr_queue, resource_queue)
            thread_list.append(process)
            process.start()

    for thread_data in thread_list:
        thread_data.join()


class MainWindow(QMainWindow):
    def __init__(self, filed_name, log_queue):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.labelsframe = {
            "left": self.ui.label,
            "top": self.ui.label_2,
            "right": self.ui.label_3,
            "front": self.ui.label_4,
            "rear": self.ui.label_5,
        }
        self.labelslane = {
            "left": self.ui.label_7,
            "top": self.ui.label_8,
            "right": self.ui.label_9,
        }

        self.camera_config = camera_model_data(filed_name)

        self.log_queue = log_queue
        self.ocr_queue = Queue()
        self.rec_queue = Queue()
        self.resource_queue = Queue()

        self.path = "2024_11_13_test_machine"
        self.channel = "01"
        self.save_thread = SaveProcessWorker(self.path, self.channel, self.log_queue)

        self.computational_resource = ComputationalResource(self.resource_queue)
        self.computational_resource.recocrylMessage.connect(self.update_rec_ocr_yl)
        self.computational_resource.frameData.connect(self.update_label_frame)
        self.computational_resource.laneData.connect(self.update_lane)

        self.original_queues = [Queue() for _ in range(len(self.camera_config))]
        self.result_queues = [Queue() for _ in range(len(self.camera_config))]

        self.image_process_workers = [RightProcessWorker(camera, result_queues, self.ocr_queue, self.resource_queue) for
                                      camera, result_queues
                                      in
                                      zip(list(self.camera_config.values())[:3], self.result_queues[:3]) if camera["direction"] in ["right"]]

        self.additional_image_workers = Process(
            target=rear_front_rec_ocr_one_process,
            args=(zip(list(self.camera_config.values())[3:], self.original_queues[3:]), self.rec_queue, self.ocr_queue, self.resource_queue, self.log_queue)
        )

        self.recognize_processes = Process(
            target=left_right_top_recognize_process,
            args=(zip(self.original_queues[:3], self.result_queues[:3]), self.log_queue,)
        )

        self.front_rear_capture = Process(
            target=front_rear_capture_process,
            args=(zip(self.camera_config.values(), self.original_queues),)
        )

        self.top_rear_capture = Process(
            target=top_left_worker_process,
            args=(zip(list(self.camera_config.values())[:3], self.result_queues[:3]), self.ocr_queue, self.resource_queue)
        )

        self.capture_processes = [RightLeftCaptureProcess(camera, original_queues, sleep_time=0.03) for
                                  camera, original_queues in
                                  zip(self.camera_config.values(), self.original_queues) if camera["direction"] in ["right", "left"]]

        self.start()

    def start(self):
        os.makedirs('tmp', exist_ok=True)
        self.delete_old_files("./tmp")

        self.save_thread.start()
        self.computational_resource.start()

        for worker in self.image_process_workers:
            worker.start()

        self.additional_image_workers.start()

        self.recognize_processes.start()

        time.sleep(5)

        self.front_rear_capture.start()

        self.top_rear_capture.start()

        for capture in self.capture_processes:
            capture.start()

    def update_lane(self, place, iscar):
        current_time = QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")
        if iscar:
            self.labelslane[place].setText(f"{place}: 车道有车 " + current_time)
            self.labelslane[place].setStyleSheet("color: red;")
        else:
            self.labelslane[place].setText(f"{place}: 车辆驶离 " + current_time)
            self.labelslane[place].setStyleSheet("color: green;")

    def update_label_frame(self, place, frame):
        # 在这里处理摄像头数据，根据摄像头标识符区分不同摄像头的数据
        if place in ("rear", "front"):
            frame = cv2.resize(frame, (582, 344))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        # 更新相应摄像头的Label
        self.labelsframe[place].setPixmap(pixmap)
        self.labelsframe[place].setAlignment(Qt.AlignCenter)

    def update_rec_ocr_yl(self, place, result):
        if place not in ("双箱", "单箱"):
            current_time = QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")
            self.ui.textEdit.insertPlainText(current_time + '\n')
        result_text = f"{place} Result: {result} \n"
        self.ui.textEdit.insertPlainText(result_text)
        self.ui.textEdit.ensureCursorVisible()

    def delete_old_files(self, path):
        try:
            files = os.listdir(path)
            # 如果文件数量大于1，则表示有旧文件需要删除
            for file in files:
                file_path = os.path.join('./tmp', file)
                os.remove(file_path)
            print("Old files deleted successfully.")
        except Exception as e:
            print("Error occurred while deleting old files:", e)


def record_logging_data(log_path, queue):
    print("log_path:", log_path, "type:", type(log_path))

    # # 获取文件夹路径
    # folder_path = os.path.dirname(log_path)
    # # 如果文件夹不存在，创建文件夹
    # if not os.path.exists(folder_path):
    #     os.makedirs(folder_path)
    #     print(f"文件夹 {folder_path} 已创建。")

    # 打开并写入文件
    with open(log_path, "w") as f:
        f.write(str(log_path) + "\n")
        while True:
            if not queue.empty():
                data = queue.get()
                f.write(str(data) + "\n")
                f.flush()
            time.sleep(0.1)


def compute_cpu_gpu_data(queue):
    cpu_list = []
    memory_list = []
    gpu_usage_list = []
    gpu_memory_list = []  # 新增用于存储 GPU 显存使用量

    while True:
        # 获取 CPU 利用率
        cpu_usage = psutil.cpu_percent(interval=1)
        cpu_list.append(cpu_usage)

        # 获取内存利用率
        memory_info = psutil.virtual_memory()
        memory_usage = memory_info.percent
        memory_list.append(memory_usage)

        # 获取 GPU 利用率和显存使用情况
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            gpu_usage = round(gpu.load * 100)  # GPU 使用率
            gpu_memory = round(gpu.memoryUsed / gpu.memoryTotal * 100)  # 显存使用率百分比
            gpu_usage_list.append(gpu_usage)
            gpu_memory_list.append(gpu_memory)

        if len(cpu_list) > 10:
            cpu_max = round(max(cpu_list), 2)
            memory_max = round(max(memory_list), 2)
            gpu_max = round(max(gpu_usage_list), 2)
            gpu_memory_max = round(max(gpu_memory_list), 2)

            # 清空列表
            cpu_list.clear()
            memory_list.clear()
            gpu_usage_list.clear()
            gpu_memory_list.clear()

            # 打印最大值
            queue.put(f"cpu:{cpu_max},gpu:{gpu_max},显存:{gpu_memory_max},内存:{memory_max}%")


if __name__ == "__main__":
    freeze_support()
    app = QApplication([])
    filed_name = str(sys.argv[1])
    # filed_name = "102806"
    # 使用示例
    folder_name = now.strftime("%Y%m%d")
    folder_filed = os.path.join(r"C:\Users\Install\Desktop\2024223\TianJinGangText", folder_name)
    os.makedirs(folder_filed, exist_ok=True)
    print(f"文件夹 '{folder_name}' 创建成功")

    # log_path = os.path.join(folder_filed, f"{filed_name}.log")
    log_path = os.path.join(folder_filed, f"{filed_name}.txt")
    if os.path.exists(log_path):
        os.remove(log_path)  # 删除文件

    queue = Queue()

    thread = threading.Thread(target=record_logging_data, args=(log_path, queue))
    thread.daemon = True
    thread.start()

    process_compute = Process(
        target=compute_cpu_gpu_data,
        args=(queue,), )
    process_compute.start()

    main_window = MainWindow(filed_name, queue)
    main_window.show()
    app.exec()
    thread.join()
    process_compute.join()
