import json
import os
os.add_dll_directory(r"C:/opencv-4.9.0/build/install/x64/vc16/bin")
os.add_dll_directory(r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin")
import cv2
import ctypes
import numpy as np
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *
from uuu import Ui_MainWindow
import threading
# from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Queue, Process, freeze_support
import test_infer as trt_infer
import infer_det_rec as det_ocr
import infer_det_rec_car as det_ocr_car
import stichImg as ts
from algorithms.detector import YOLOv5Detector
import Container_det_trt_yolov5 as cont_trt_infer
import time, glob, socket, zmq
from datetime import datetime
from configs import config_5s_trt as my_config
import copy
from clear_img import clean_all_directories

from reader_data.logger_cpu_gpu import cpu_info, gpu_info

weights_dict = {
    "ocr_det_config": "./config_car/det/my_car_det_r50_db++_td_tr.yml",
    "ocr_rec_config": "./config_car/rec/my_rec_chinese_lite_train_v2.0.yml"
}
lp = det_ocr_car.OCR_process(weights_dict)



def process_frames(frames, id):
    print(f"拼图的图片数量为: {len(frames)}")
    start_time = time.time()
    print(id, "拼图的开始时间:", start_time)

    print(f"开始拼接 ID: {id}")
    my_stiching = ts.stiching_img()
    result = None
    if id == 1:
        result = my_stiching.stiching(frames, "top")
        # cv2.imwrite(f"pingtu/{id}.jpg", result)
    elif id == 0:
        result = my_stiching.stiching(frames, "left")
        # cv2.imwrite(f"pingtu/{id}.jpg", result)
    elif id == 2:
        result = my_stiching.stiching(frames, "right")
        # cv2.imwrite(f"pingtu/{id}.jpg", result)
    print(f"拼接完成 ID: {id}")

    # result = cv2.imread(f"pingtu/{id}.jpg")
    print(f"拼图的时间为: {time.time() - start_time}")

    cv2.imwrite(f'tmp/{id}.jpg', result)

    # print(f"进程 {id} 完成后继续执行任务")


def rec_time(frames):
    for frame in frames:
        start_time = time.time()
        result = lp.process_imgs([frame])
        # print("时间:",time.time()-start_time)
        # print(result)


def weitr_img(data):
    folder_path = fr"C:\TianJinGangTest\{data}"
    jpg_files = get_all_jpg_files(folder_path)
    print(f"找到的图片文件: {jpg_files}")
    frames_list = display_images(jpg_files)
    print(f"总共找到的帧数: {len(frames_list)}")
    return frames_list


def get_all_jpg_files(folder_path):
    return glob.glob(os.path.join(folder_path, "*.jpg")) + glob.glob(os.path.join(folder_path, "*.jpeg"))


def display_images(image_paths):
    frames_list = []
    for image_path in image_paths:
        img = cv2.imread(image_path)
        frames_list.append(img)
    return frames_list


def run():
    # base_directory = 'C:/TianjinGangTest'
    # clean_all_directories(base_directory, keep_latest=0)

    # frames_to_process_0 = self.weitr_img(0)
    frames_to_process_0 = weitr_img(0)
    frames_to_process_1 = weitr_img(1)
    frames_to_process_2 = weitr_img(2)
    # frames_to_process_2 = self.weitr_img(2)
    # self.process_frames(frames_to_process, self.camera_id)
    # threading.Thread(target=self.process_frames, args=(frames_to_process_0, 0)).start()
    # get_gpu_info()
    # threading.Thread(target=process_frames, args=(frames_to_process_1, 1)).start()
    p = threading.Thread(target=process_frames, args=(frames_to_process_0, 0))
    p.start()
    p.join()

    a = threading.Thread(target=process_frames, args=(frames_to_process_1, 1))
    a.start()
    a.join()

    c = threading.Thread(target=process_frames, args=(frames_to_process_2, 2))
    # threading.Thread(target=self.process_frames, args=(frames_to_process_2, 2)).start()
    c.start()
    c.join()


if __name__ == '__main__':
    run()
