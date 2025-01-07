# -*- coding: utf-8 -*-
import json
import os
import sys
import cv2
import ctypes
import numpy as np
print("cv2:", cv2.__file__)
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
import time, socket, zmq
from datetime import datetime
from configs import config_5s_trt as my_config
import copy
import os

# 获取当前文件所在目录的绝对路径
filed_abspath = os.path.dirname(os.path.abspath(__file__))
print("当前文件夹执行在路径:", filed_abspath)

# 正确拼接 ocr_det_config 的路径
ocr_det_config = os.path.join(filed_abspath, "config", "det", "my_det_r50_db++_td_tr.yml")
print("ocr_det_config:", ocr_det_config)

# 当前exe文件在路径
oring_path = sys.argv[0]
real_path = os.path.realpath(oring_path)
filed_path = os.path.dirname(real_path)
save_path = os.path.join(filed_path, "tmp")

xiangti = ''


class ImageCaptureProcess(Process):
    def __init__(self, camera_info, queue):
        super().__init__()
        self.camera_id = camera_info['id']
        self.file_path = camera_info['file_path']
        self.direction = camera_info['direction']
        self.sleep_time = camera_info['sleep_time']
        self.roi = camera_info.get('roi', None)
        self.queue = queue
        self.front_rear = ("front", "rear")
        self.left_right = ("right", "left")
        self.dsize = (900, 600)
        self.roi_cols = (self.roi[2], self.roi[3])
        self.roi_rows = (self.roi[0], self.roi[1])
        self.front_rear_count = 0
        self.counter = 0

    def run(self):
        try:
            cap = cv2.VideoCapture(self.file_path, cv2.CAP_FFMPEG, [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY])
            cap.set(cv2.CAP_PROP_FPS, 12)
            while True:
                time.sleep(self.sleep_time)
                cap.grab()
                ret, frame = cap.retrieve()
                if not ret:
                    self.counter += 1
                    if self.counter >= 5:
                        cap.release()
                        cap = cv2.VideoCapture(self.file_path, cv2.CAP_FFMPEG, [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY])
                        self.counter = 0
                    continue

                if self.direction in self.front_rear:
                    frame_crop = frame[self.roi_rows[0]:self.roi_rows[1], self.roi_cols[0]:self.roi_cols[1]]
                    frame_cpu = frame_crop[:, :, :3]  # 裁剪通道
                    self.queue.put((self.camera_id, frame_cpu))
                    self.front_rear_count = 0
                else:
                    frame = frame[:, :, :3]
                    self.queue.put((self.camera_id, frame, [self.roi_rows[0], self.roi_rows[1], self.roi_cols[0], self.roi_cols[1]]))
                self.counter = 0
        except Exception as error:
            print(f"ImageCaptureProcess---{self.direction}:图片读取有问题:{error}")


class ImageProcessWorker(QThread):
    frameCaptured = Signal(int, int)
    image_processed = Signal(int, object)
    dataQueued = Signal(int, object)

    def __init__(self, camera_info, result_queue, qu2):
        super().__init__()
        self.camera_id = camera_info['id']
        self.file_path = camera_info['file_path']
        self.direction = camera_info['direction']
        self.roi = camera_info.get('roi', None)
        self.result_queue = result_queue
        self.ocr_queue = qu2
        self.true_threshold = 3
        self.detector = None
        if self.direction == "right":
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
            self.image_processed.emit(20, None)
            xiangti = '双箱'
        else:
            self.image_processed.emit(21, None)
            xiangti = '单箱'

    def process_frames(self, frames, id, num, status):
        if id == 2 and self.direction == "right":
            result, state = self.my_stiching.stiching(frames, "right", num, status)
        elif id == 1:
            result, state = self.my_stiching.stiching(frames, "top", num, status)
        else:
            result, state = self.my_stiching.stiching(frames, "left", num, status)

        if state:
            if id == 2:
                self.executor.submit(self.call_shuangxiang, result)
            resized_image = cv2.resize(result, (1278, 344))
            self.dataQueued.emit(id, resized_image)
            cv2.imwrite(os.path.join(save_path, f'{id}.jpg'), resized_image)


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
                if result == 1:
                    self.consecutive_zero_count = 0
                    self.consecutive_true_count += 1
                    if self.consecutive_true_count >= self.true_threshold:
                        # 连续无车
                        self.consecutive_true_count = 0
                        if len(self.frames_to_process) < 5:
                            self.frames_to_process = []
                        # 标志位，代表车辆是否有进入记录
                        if self.car_in:
                            self.car_in = False
                            self.frameCaptured.emit(self.camera_id, 0)
                            self.count += 1
                            self.count_all += len(self.frames_to_process)

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
                        self.frameCaptured.emit(self.camera_id, 1)
                        self.image_processed.emit(self.camera_id, None)
                if result == "NO" and self.car_in:
                    self.frames_to_process.append(frame)


class ImageProcessRecognize(Process):
    def __init__(self, original_queue, result_queue):
        super().__init__()
        self.original_queue = original_queue
        self.result_queue = result_queue
        self.count = 0
        self.count_top = 0

    def run(self):
        self.trt_infer = trt_infer
        cfg_dict = {
            "engine_path": os.path.join(filed_path, "tianJinGang.engine"),
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
                camera_id, frame, seg_data = self.original_queue.get()
                self.count += 1
                self.count_top += 1
                if self.count % 2 == 0 and camera_id != 1:
                    seg_frame = frame[seg_data[0]:seg_data[1], seg_data[2]:seg_data[3]]
                    thread1 = self.trt_infer.inferThread(self.mobilenet_wrapper, [seg_frame])
                    thread1.start()
                    thread1.join()
                    result = thread1.get_result()
                    self.result_queue.put((camera_id, result, frame))
                    self.count = 0
                else:
                    if camera_id != 1:
                        self.result_queue.put((camera_id, "NO", frame))
                if self.count_top % 5 == 0 and camera_id == 1:
                    seg_frame = frame[seg_data[0]:seg_data[1], seg_data[2]:seg_data[3]]
                    thread1 = self.trt_infer.inferThread(self.mobilenet_wrapper, [seg_frame])
                    thread1.start()
                    thread1.join()
                    result = thread1.get_result()
                    self.result_queue.put((camera_id, result, frame))
                    self.count_top = 0


class ImageProcessWorker2(QThread):
    image_processed = Signal(int, object)
    dataQueued = Signal(int, object)

    def __init__(self, camera_info, qu1, qu2, qu3):
        super().__init__()
        self.camera_id = camera_info['id']
        self.file_path = camera_info['file_path']
        self.direction = camera_info['direction']
        self.roi = camera_info.get('roi', None)
        self.queue = qu1
        self.rec_queue = qu2
        self.ocr_queue = qu3
        self.frame_counter = 0
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.initialize_inference()

    def initialize_inference(self):
        PLUGIN_LIBRARY = os.path.join(filed_path, "myplugins.dll")
        engine_file_path = os.path.join(filed_path, "truck.engine")
        ctypes.CDLL(PLUGIN_LIBRARY)
        self.csd_detector = cont_trt_infer.CSD_Detector(engine_file_path)  # 初始化detector
        self.my_container_detect = cont_trt_infer.container_detect(self.csd_detector)

    def delete_old_files(self):
        try:
            # 获取tmp文件夹下的所有文件
            files = os.listdir(save_path)
            # 如果文件数量大于1，则表示有旧文件需要删除
            for file in files:
                file_path = os.path.join(save_path, file)
                os.remove(file_path)
        except Exception as e:
            print("Error occurred while deleting old files:", e)

    def run(self):
        res_dict_lst = []
        while True:
            if not self.queue.empty():
                self.camera_id, frame = self.queue.get()
                self.frame_counter += 1
                # rec队列采集 front==0
                if self.camera_id == 0:
                    if self.frame_counter == 3:
                        self.rec_queue.put((self.camera_id, frame))
                        self.frame_counter = 0
                # suanfa_time = time.time()
                self.my_container_detect.process_frame(frame)
                _, reuslt_dict = self.my_container_detect.get_result()
                if reuslt_dict:
                    res_dict_lst.append(reuslt_dict)
                if self.my_container_detect.non_truck > 5 and self.my_container_detect.new_truck:
                    # 当连续12帧(约1s)没有集装箱面，且之前有卡车进入时，获取前一段时间面积最大帧
                    reuslt_dict, _ = self.my_container_detect.get_result()
                    self.executor.submit(self.clear_writer_img, self.camera_id, reuslt_dict['img'])
                    # ocr队列采集
                    self.ocr_queue.put(((self.camera_id + 6), reuslt_dict['img']))
                    final_label, _ = self.my_container_detect.door_label_vote(res_dict_lst)
                    # 界面刷新
                    self.executor.submit(self.dataQueued_show, self.camera_id, reuslt_dict['img'])
                    if final_label == 'door':
                        self.ocr_queue.put(((self.camera_id + 10), None))
                    if self.camera_id == 1:
                        self.rec_queue.put((5, None))
                        self.ocr_queue.put((5, None))
                    self.my_container_detect.new_truck = False
                    self.my_container_detect.max_area_dict.clear()
                    self.my_container_detect.res_dict.clear()
                    res_dict_lst.clear()

    def clear_writer_img(self, camera_id, frame):
        file_path = os.path.join(save_path, f'{camera_id + 3}.jpg')
        if camera_id == 0:
            self.delete_old_files()
        cv2.imwrite(file_path, frame)

    def dataQueued_show(self, camera_id, frame):
        self.dataQueued.emit(camera_id, frame)


class SaveProcessWorker(QThread):
    def __init__(self, path, channel):
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
        self.executor = ThreadPoolExecutor(max_workers=6)

    def rename_file(self, file, output_path):
        max_retries = 3
        retries = 0
        file_path = os.path.join(save_path, file)
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
        current_time = datetime.now().strftime("%H%M%S")
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

        output_folder = os.path.join(check_floder, self.vehicle_count, "Transfer")
        os.makedirs(output_folder, exist_ok=True)
        self.check_and_save(current_date, current_time, output_folder)

    def check_and_save(self, current_date, current_time, output_folder):
        position = ["left", "top", "right", "Front", "rear"]
        li = []
        for index, tmp_data in enumerate(position):
            # 图的名称
            filename = f"{self.ch}_{current_date}_{current_time}_{self.vehicle_count}_Transfer_{tmp_data}_{self.image_number}.jpg"
            # 每张图的最终路径
            output_path = os.path.join(output_folder, filename)
            self.executor.submit(self.rename_file, f'{index}.jpg', output_path)
            li.append(output_path)
        self.executor.submit(self.rename_file, "ocr.txt", os.path.join(output_folder, "ocr.txt"))
        self.executor.submit(self.rename_file, "rec.txt", os.path.join(output_folder, "rec.txt"))
        if os.path.exists(os.path.join(save_path, f'chepai.jpg')):
            self.executor.submit(self.rename_file, "chepai.jpg", os.path.join(output_folder, "chepai.jpg"))

        try:
            global xiangti
            if xiangti == "":
                time.sleep(3)
                self.executor.submit(self.save_xiangti, output_folder, xiangti, li)
            xiangti = ''
        except Exception as e:
            # 在这里处理异常
            print("An error occurred:", e)

    def save_xiangti(self, output_folder, xiangti, li):
        ocr_path = os.path.join(output_folder, "ocr.txt")
        with open(ocr_path, 'a', encoding='utf-8') as file:
            file.write(xiangti + '\n')
        self.send_zmq_string(li)

    def send_zmq_string(self, li):
        # 获取当前时间
        current_time = datetime.now()
        # 格式化时间戳
        timestamp = current_time.strftime("%Y%m%d%H%M%S")
        # 构建数据字符串
        data = f"[C|{timestamp}|{self.ch}|{self.vehicle_count}|{li[3]}|{li[4]}|{li[0]}|{li[2]}|{li[1]}|{self.ip_address}|D]"
        # 发布数据
        self.publisher.send_string(data)

    def check_files_existence(self, tmp_path):
        # 要检查的文件列表
        files_to_check = ['0.jpg', '1.jpg', '2.jpg', '3.jpg', '4.jpg', 'ocr.txt', 'rec.txt']
        # 检查每个文件是否存在
        for file_name in files_to_check:
            file_path = os.path.join(tmp_path, file_name)
            if not os.path.exists(file_path):
                return False
        return True

    def run(self):
        while True:
            if self.check_files_existence(save_path):
                self.check()
            time.sleep(1)


class OcrThread(QThread):
    ocrMessage = Signal(int, list)

    def __init__(self, queue):
        QThread.__init__(self)
        self.queue = queue
        self.config_dict = {
            "ocr_det_config": os.path.join(filed_abspath, "config", "det", "my_det_r50_db++_td_tr.yml"),
            "ocr_rec_config": os.path.join(filed_abspath, "config", "rec", "my_en_PP-OCRv3_rec.yml")
        }
        self.ocr = det_ocr.OCR_process(self.config_dict)
        self.executor = ThreadPoolExecutor(max_workers=1)

    def write_frame(self, path, frame):
        cv2.imwrite(path, frame)

    def run(self):
        ocr_data_dict = {"front": "", "rear": ""}
        result_list = []
        result = None
        while True:
            id, frame = self.queue.get()
            if id == 10:
                ocr_data_dict["front"] = "箱门朝前"
                self.ocrMessage.emit(1, "箱门朝前")
            if id == 11:
                ocr_data_dict["rear"] = "箱门朝后"
                self.ocrMessage.emit(1, "箱门朝后")
            if id != 5:
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
            if id == 5:
                ultra_result_one = det_ocr.vote2res(result_list)
                result_list = []
                self.write_ocr_data(ultra_result_one, copy.deepcopy(ocr_data_dict))
                self.ocrMessage.emit(1, ultra_result_one)
                ocr_data_dict = {"front": "", "rear": ""}
            time.sleep(0.001)

    def write_ocr_data(self, ultra_result, ocr_data_dict):
        file_path = os.path.join(save_path, 'ocr.txt')
        # 写入结果到文件中
        with open(file_path, 'w', encoding='utf-8') as file:
            ultra_result = ultra_result[:-4] + ' ' + ultra_result[-4:]
            file.write(ultra_result + '\n')
            if ocr_data_dict['front']:
                file.write(ocr_data_dict['front'] + '\n')
            if ocr_data_dict['rear']:
                file.write(ocr_data_dict['rear'] + '\n')


class RecThread(QThread):
    recMessage = Signal(int, list)

    def __init__(self, queue):
        QThread.__init__(self)

        self.weights_dict = {
            "ocr_det_config": os.path.join(filed_abspath, "config_car", "det", "my_car_det_r50_db++_td_tr.yml"),
            "ocr_rec_config": os.path.join(filed_abspath, "config_car", "rec", "my_rec_chinese_lite_train_v2.0.yml"),
        }
        self.lp = det_ocr_car.OCR_process(self.weights_dict)
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.queue = queue
        self.cont_num = 1

    def run(self):
        result_list = []
        save_frame = []

        while True:
            id, frame = self.queue.get()
            if id != 5:
                use_frame = cv2.GaussianBlur(frame, (5, 5), 0)
                result = self.lp.process_imgs([use_frame])
                if len(result) > 0:
                    result_list.append([result, frame])
                    save_frame.append(frame)

            if id == 5:
                ultra_result, save_frame_idx = det_ocr_car.get_finalResult(result_list)
                if len(save_frame) > 0:
                    final_save = save_frame[save_frame_idx]
                    self.executor.submit(self.writer_img, os.path.join(save_path, 'chepai.jpg'), final_save)
                result_list = []
                save_frame = []
                self.executor.submit(self.writer_rec_data, ultra_result)

    def writer_rec_data(self, ultra_result):
        file_path = os.path.join(save_path, 'rec.txt')
        self.recMessage.emit(1, ultra_result)
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(ultra_result)

    def writer_img(self, file_path, frame):
        cv2.imwrite(file_path, frame)


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.labels1 = [self.ui.label, self.ui.label_2, self.ui.label_3]
        self.labels2 = [self.ui.label_4, self.ui.label_5]
        self.labels = [self.ui.label_7, self.ui.label_8, self.ui.label_9]

        os.makedirs(save_path, exist_ok=True)

        camera_model_path = os.path.join(filed_path, 'camera_model.json')
        with open(camera_model_path, 'r') as f:
            self.camera_config = json.load(f)

        self.ocr_queue = Queue()
        self.rec_queue = Queue()

        self.ocr_thread = OcrThread(self.ocr_queue)
        self.ocr_thread.ocrMessage.connect(self.update_ocr)

        self.rec_thread = RecThread(self.rec_queue)
        self.rec_thread.recMessage.connect(self.update_rec)

        self.path = os.path.join(filed_path, '1108_save_test')
        os.makedirs(self.path, exist_ok=True)

        self.channel = "01"
        os.makedirs(os.path.join(self.path, self.channel), exist_ok=True)
        self.save_thread = SaveProcessWorker(self.path, self.channel)

        self.original_queues = [Queue() for _ in range(len(self.camera_config))]
        self.result_queues = [Queue() for _ in range(len(self.camera_config))]

        self.image_process_workers = [ImageProcessWorker(camera_info, result_queues, self.ocr_queue) for
                                      camera_info, result_queues
                                      in
                                      zip(list(self.camera_config.values())[:3], self.result_queues[:3])]

        for worker in self.image_process_workers:
            worker.image_processed.connect(self.handle_image_processed)
            worker.dataQueued.connect(self.update_label_1)
            worker.frameCaptured.connect(self.update_label)

        self.additional_image_workers = [ImageProcessWorker2(camera_info, queue, self.rec_queue, self.ocr_queue) for
                                         camera_info, queue in
                                         zip(list(self.camera_config.values())[3:], self.original_queues[3:])]

        for worker in self.additional_image_workers:
            worker.image_processed.connect(self.handle_image_processed)
            worker.dataQueued.connect(self.update_label_2)

        self.recognize_processes = [ImageProcessRecognize(original_queues, result_queues) for
                                    original_queues, result_queues in
                                    zip(self.original_queues[:3], self.result_queues[:3])]

        self.capture_processes = [ImageCaptureProcess(camera_info, original_queues) for camera_info, original_queues in
                                  zip(self.camera_config.values(), self.original_queues)]

        self.start()

    def start(self):

        os.makedirs(save_path, exist_ok=True)

        self.ocr_thread.start()
        self.rec_thread.start()
        self.save_thread.start()
        print("1111111111111111111111111111111:")

        for worker in self.image_process_workers:
            worker.start()

        for worker in self.additional_image_workers:
            worker.start()

        for recognize in self.recognize_processes:
            recognize.start()

        time.sleep(15)

        for process in self.capture_processes:
            process.start()

    def update_label(self, camera_id, iscar):
        current_time = QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")
        if iscar:
            self.labels[camera_id].setText("result: 车道有车 " + current_time)
            self.labels[camera_id].setStyleSheet("color: red;")
        else:
            self.labels[camera_id].setText("result: 车辆驶离 " + current_time)
            self.labels[camera_id].setStyleSheet("color: green;")

    def update_label_1(self, camera_id, frame):
        # 在这里处理摄像头数据，根据摄像头标识符区分不同摄像头的数据
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        # 更新相应摄像头的Label
        self.labels1[camera_id].setPixmap(pixmap)
        self.labels1[camera_id].setAlignment(Qt.AlignCenter)

    def update_label_2(self, camera_id, frame):
        # 在这里处理摄像头数据，根据摄像头标识符区分不同摄像头的数据
        frame = cv2.resize(frame, (582, 344))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        # 更新相应摄像头的Label
        self.labels2[camera_id].setPixmap(pixmap)
        self.labels2[camera_id].setAlignment(Qt.AlignCenter)

    def update_ocr(self, camera_id, sorted_data):
        current_time = QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")
        self.ui.textEdit.insertPlainText(current_time + '\n')
        result_text = f"Ocr Result: {sorted_data} \n"
        self.ui.textEdit.insertPlainText(result_text)
        self.ui.textEdit.ensureCursorVisible()

    def update_rec(self, camera_id, result):
        current_time = QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")
        self.ui.textEdit.insertPlainText(current_time + '\n')
        result_text = f"Rec Result: {result} \n"
        self.ui.textEdit.insertPlainText(result_text)
        self.ui.textEdit.ensureCursorVisible()

    def handle_image_processed(self, camera_id, image):
        # 处理已处理的图像
        if camera_id == 20:
            result_text = f"Yl Result: 双箱 \n"
            self.ui.textEdit.insertPlainText(result_text)
            self.ui.textEdit.ensureCursorVisible()

        if camera_id == 21:
            result_text = f"Yl Result: 单箱 \n"
            self.ui.textEdit.insertPlainText(result_text)
            self.ui.textEdit.ensureCursorVisible()


if __name__ == "__main__":
    freeze_support()
    app = QApplication([])
    main_window = MainWindow()
    main_window.show()
    app.exec()
