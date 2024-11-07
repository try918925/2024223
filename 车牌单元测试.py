# -*- coding: utf-8 -*-
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
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Queue, Process, freeze_support
import test_infer as trt_infer
import infer_det_rec as det_ocr
import infer_det_rec_car as det_ocr_car
import stichImg as ts
from algorithms.detector import YOLOv5Detector
import Container_det as cont_trt_infer
# import Container_det_trt_yolov5 as cont_trt_infer
import time, glob, socket, zmq
from datetime import datetime
from configs import config_5s_trt as my_config
from configs import config_5s_trt as door_config
import copy

config_dict = {
    "ocr_det_config": "./config/det/my_det_r50_db++_td_tr.yml",
    "ocr_rec_config": "./config/rec/my_en_PP-OCRv3_rec.yml"
}
ocr = det_ocr.OCR_process(config_dict)

weights_dict = {
    "ocr_det_config": "./config_car/det/my_car_det_r50_db++_td_tr.yml",
    "ocr_rec_config": "./config_car/rec/my_rec_chinese_lite_train_v2.0.yml"
}
lp = det_ocr_car.OCR_process(weights_dict)

decoder = cv2.cudacodec.createVideoReader("C:/TianJinGangTest/25tup/top.mp4")


while True:
    ret, frame = decoder.nextFrame()
    if not ret or frame is None:
        continue

    use_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    result = lp.process_imgs([use_frame])
    if len(result) > 0:
        result_list.append([result, frame])
        save_frame.append(frame)

        ultra_result, save_frame_idx = det_ocr_car.get_finalResult(result_list)
    if len(save_frame) > 0:
        final_save = save_frame[save_frame_idx]
        writer_img('tmp/chepai.jpg', final_save)
    result_list = []
    save_frame = []
    time.sleep(0.001)
