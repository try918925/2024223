import json
import os
os.add_dll_directory(r"C:/opencv_build_data/opencv-4.9.0/build/install/x64/vc16/bin")
os.add_dll_directory(r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/bin")
import cv2
import ctypes
import numpy as np
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *
from uuu import Ui_MainWindow
import threading
import multiprocessing
from multiprocessing import Pool, Queue, Process, Lock
import test_infer as trt_infer
import infer_det_rec as det_ocr
import infer_det_rec_car as det_ocr_car
import stichImg as ts
from algorithms.detector import YOLOv5Detector
# import Container_det as cont_trt_infer
import Container_det_trt_yolov5 as cont_trt_infer
import time
from datetime import datetime
import logging

# 配置日志记录器
# logging.basicConfig(filename='log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(filename='log.txt', filemode='a', encoding='utf-8', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
# True = True
xiangti = ''
xiangti_count = 0

rear_truck = 'ok'
front_truck = 'no'
front_img = None
rear_img = ""


# front_count = 0
# rear_count = 0

def time_str(fmt=None):
    if fmt is None:
        fmt = '%Y_%m_%d_%H_%M_%S'
    return datetime.today().strftime(fmt)



class ImageCaptureProcess(Process):
    def __init__(self, camera_info, queue):
        super().__init__()
        self.camera_id = camera_info['id']
        self.file_path = camera_info['file_path']
        self.roi = camera_info.get('roi', None)
        self.queue = queue

    def run(self):
        global run_Flag
        decoder = cv2.cudacodec.createVideoReader(self.file_path)
        try:
            frame_counter = 0
            while run_Flag:
                ret, frame = decoder.nextFrame()
                if not ret or frame is None:
                    continue
                frame = frame.download()
                # print(frame.shape)
                frame = frame[:, :, :3]
                frame_counter += 1
                if frame_counter % 2 == 1:
                    self.queue.put((self.camera_id, frame))
        except Exception as e:
            print("Error:", e)

    # def quit(self):
    #     import sys
    #     # Clean up resources and exit the process
    #     print("Process", self.camera_id, "exiting...")
    #     #self.queue.close()
    #     sys.exit(0)



class ImageProcessWorker(QThread):
    frameCaptured = Signal(int, int)
    image_processed = Signal(int, object)
    dataQueued = Signal(int, object)

    def __init__(self, camera_info, qu1, qu2):
        super().__init__()
        self.camera_id = camera_info['id']
        self.file_path = camera_info['file_path']
        self.roi = camera_info.get('roi', None)
        self.result = camera_info['direction']
        self.queue = qu1
        self.ocr_queue = qu2

        self.consecutive_true_count = 0
        self.true_threshold = 3
        self.initialize_inference()

    def initialize_inference(self):
        cfg_dict = {
            "engine_path": "./tianJinGang.engine",
            "class": [0, 1]  # 0 有车，result：1没车
        }
        self.mobilenet_wrapper = trt_infer.MobilenetTRT(cfg_dict["engine_path"])
        # 执行 warm-up 操作
        for i in range(10):
            # create a new thread to do warm_up
            thread1 = trt_infer.warmUpThread(self.mobilenet_wrapper)
            thread1.start()
            thread1.join()

    def call_shuangxiang(self, frame):
        from configs import config_5s_trt as my_config
        detector = YOLOv5Detector.from_config(my_config)
        global xiangti
        img = np.ascontiguousarray(frame)
        # 保证数据和训练数据尺寸统一
        img = cv2.resize(img, (4160, 1040))
        # for _ in range(500):
        tik = time.time()
        obj_list = []
        obj_list = detector.det([img])[0]
        print(obj_list, time.time() - tik)
        print(len(obj_list))
        if len(obj_list) == 2:
            self.image_processed.emit(20, None)
            xiangti = '双箱'
            return 2
            # print('双箱')
        else:
            self.image_processed.emit(21, None)
            xiangti = '单箱'
            return 1
            # print('单箱')

    def process_frames(self, frames, id):
        my_stiching = ts.stiching_img()
        if id == 1:
            result = my_stiching.stiching(frames, "top")
        if id == 0:
            result = my_stiching.stiching(frames, "left")
        if id == 2:
            result = my_stiching.stiching(frames, "right")
            thread = threading.Thread(target=self.call_shuangxiang, args=(result,))
            # 启动线程
            thread.start()

        file_path = f'tmp/{id}.jpg'
        # 保存图片
        cv2.imwrite(file_path, result)

        resized_image = cv2.resize(result, (1278, 344))
        self.dataQueued.emit(id, resized_image)
        # path = f"my_test_img/pingtu/{time_str()}.jpg"
        # grabber = threading.Thread(target=self.writer_img, args=(path, result))
        # grabber.start()

    def writer_img(self, path, frame):
        cv2.imwrite(path, frame)
        # print(f"-------------{self.result}---方向图片写入完成--------------")

    #
    def run(self):
        self.frames_to_process = []
        self.car_in = False
        consecutive_zero_count = 0
        ocr_count = 0
        while True:
            if not self.queue.empty():
                camera_id, frame = self.queue.get()
                print(f"--------------ImageProcessWorker_left_right_top-----------------{self.queue.qsize()}")
                seg_frame = frame[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]

                # path = f"my_test_img/{self.result}/{time_str()}.jpg"
                # grabber = threading.Thread(target=self.writer_img, args=(path, frame))
                # grabber.start()

                start_time = time.time()
                thread1 = trt_infer.inferThread(self.mobilenet_wrapper, [seg_frame])
                thread1.start()
                thread1.join()
                result = thread1.get_result()
                end_time = time.time()
                print(f"--left,right,top 识别有无车时间: {end_time - start_time:.6f} 秒",result)


                if result == 1:
                    consecutive_zero_count = 0
                    self.consecutive_true_count += 1
                    if self.consecutive_true_count >= self.true_threshold:
                        # 连续无车
                        self.consecutive_true_count = 0
                        # if len(self.frames_to_process) < 3:
                        #     self.frames_to_process = []
                        # 标志位，代表车辆是否有进入记录
                        if self.car_in and self.frames_to_process:
                            # consecutive_zero_count = 0
                            print("车辆驶离")
                            self.car_in = False
                            self.frameCaptured.emit(self.camera_id, 0)
                            # if self.camera_id ==0:
                            #     self.ocr_queue.put([5,None])

                            if self.camera_id != 1:
                                self.ocr_queue.put([5, None])

                            threading.Thread(target=self.process_frames,
                                             args=(self.frames_to_process.copy(), self.camera_id)).start()
                            self.frames_to_process = []

                if result == 0:
                    self.consecutive_true_count = 0
                    consecutive_zero_count += 1
                    self.frames_to_process.append(frame)
                    # ocr队列
                    # if (consecutive_zero_count) % 2 == 0:
                    # top不采集
                    if self.camera_id != 1 and ocr_count==3:
                        self.ocr_queue.put([self.camera_id, frame])
                        ocr_count = 0
                    # 阈值

                    # path = f"my_test_img/'have_'+{self.result}/{time_str()}.jpg"
                    # grabber = threading.Thread(target=self.writer_img, args=(path, frame))
                    # grabber.start()
                    # print(f"=========={self.result} :有车=========")

                    if consecutive_zero_count > 3:
                        self.car_in = True
                        self.frameCaptured.emit(self.camera_id, 1)
                        self.image_processed.emit(camera_id, frame)


class ImageProcessWorkerFront(QThread):
    image_processed = Signal(int, object)
    dataQueued = Signal(int, object)

    def __init__(self, camera_info, qu1, qu2, qu3):
        super().__init__()
        self.camera_id = camera_info['id']
        self.file_path = camera_info['file_path']
        self.roi = camera_info.get('roi', None)
        self.result = camera_info['direction']
        self.queue = qu1
        self.rec_queue = qu2
        self.ocr_queue = qu3
        self.frame_counter = 0
        self.initialize_inference()

    def initialize_inference(self):
        PLUGIN_LIBRARY = "./myplugins.dll"
        engine_file_path = "truck_old.engine"
        ctypes.CDLL(PLUGIN_LIBRARY)
        self.csd_detector = cont_trt_infer.CSD_Detector(engine_file_path)  # 初始化detector
        self.my_container_detect = cont_trt_infer.container_detect(self.csd_detector)

    def delete_old_files(self):
        import glob
        import shutil
        try:
            # 获取tmp文件夹下的所有文件
            files = glob.glob('tmp/*')
            # 如果文件数量大于1，则表示有旧文件需要删除
            if len(files) > 1:
                # 获取最新保存的文件
                # latest_file = max(files, key=os.path.getctime)
                # 删除除最新保存的文件之外的其他文件
                for file in files:
                    # if file != latest_file:
                    os.remove(file)
                print("Old files deleted successfully.")
            else:
                print("No old files to delete.")
        except Exception as e:
            print("Error occurred while deleting old files:", e)

    def writer_img(self, path, frame):
        cv2.imwrite(path, frame)
        # print("----------------------完成--------------")

    def run(self):
        global front_img
        print(
            f"------------------ImageProcessWorker2---id:{self.camera_id},roi:{self.roi},file_path:{self.file_path}------------------执行")
        res_dict_lst = []
        front_count = 0
        while True:
            if not self.queue.empty():
                print(f"--------------ImageProcessWorkerFront-----------------{self.queue.qsize()}")
                camera_id, frame = self.queue.get()
                frame = frame[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]

                # path = f"my_test_img/{self.result}/{time_str()}.jpg"
                # grabber = threading.Thread(target=self.writer_img, args=(path, frame))
                # grabber.start()

                self.frame_counter += 1
                # rec队列采集
                if self.frame_counter == 3:
                    self.rec_queue.put([self.camera_id, frame])
                    self.frame_counter = 0

                start_time = time.time()
                self.my_container_detect.process_frame(frame)
                _, reuslt_dict = self.my_container_detect.get_result()
                end_time = time.time()
                print(f"------前摄像头识别时间: {end_time - start_time:.6f} 秒")


                if reuslt_dict:
                    print("--------------front----识别成功---------------")
                    res_dict_lst.append(reuslt_dict)
                    front_count += 1
                    if front_count >= 5:
                        if rear_truck == "no":
                            self.dataQueued.emit(self.camera_id, reuslt_dict['img'])
                        if rear_truck == "ok":
                            front_img = reuslt_dict['img']
                else:
                    front_count = 0

                if self.my_container_detect.non_truck > 5 and self.my_container_detect.new_truck:
                    # 当连续10帧(约1s)没有集装箱面，且之前有卡车进入时，获取前一段时间面积最大帧
                    reuslt_dict, _ = self.my_container_detect.get_result()
                    print(reuslt_dict['area'])
                    # ocr队列采集
                    self.ocr_queue.put([(self.camera_id + 6), reuslt_dict['img']])
                    print("----------------------传入数据---ocr_queue----")

                    final_label, showImg = self.my_container_detect.door_label_vote(res_dict_lst)

                    print("当连续10帧(约1s)没有集装箱面，且之前有卡车进入时，获取前一段时间面积最大帧", final_label)
                    if final_label == 'door':
                        # print(f"final_label == 'door': self.camera_id:{self.camera_id}")
                        self.ocr_queue.put([(self.camera_id + 10), None])
                        print("----------------------传入数据---ocr_queue----")
                    else:
                        self.ocr_queue.put([(self.camera_id + 20), None])

                    file_path = f'tmp/{self.camera_id + 3}.jpg'
                    self.delete_old_files()
                    logging.info('New car move in ')

                    cv2.imwrite(file_path, reuslt_dict["img"])
                    # cv2.imwrite(file_path, showImg)

                    self.my_container_detect.new_truck = False
                    self.my_container_detect.max_area_dict.clear()
                    self.my_container_detect.res_dict.clear()
                    res_dict_lst.clear()


class ImageProcessWorkerRear(QThread):
    image_processed = Signal(int, object)
    dataQueued = Signal(int, object)

    def __init__(self, camera_info, qu1, qu2, qu3):
        super().__init__()
        self.camera_id = camera_info['id']
        self.file_path = camera_info['file_path']
        self.roi = camera_info.get('roi', None)
        self.result = camera_info['direction']
        self.queue = qu1
        self.rec_queue = qu2
        self.ocr_queue = qu3
        self.frame_counter = 0
        self.initialize_inference()

    def initialize_inference(self):
        PLUGIN_LIBRARY = "./myplugins.dll"
        engine_file_path = "truck_old.engine"
        ctypes.CDLL(PLUGIN_LIBRARY)
        self.csd_detector = cont_trt_infer.CSD_Detector(engine_file_path)  # 初始化detector
        self.my_container_detect = cont_trt_infer.container_detect(self.csd_detector)

    def delete_old_files(self):
        import glob
        import shutil
        try:
            # 获取tmp文件夹下的所有文件
            files = glob.glob('tmp/*')
            # 如果文件数量大于1，则表示有旧文件需要删除
            if len(files) > 1:
                # 获取最新保存的文件
                # latest_file = max(files, key=os.path.getctime)
                # 删除除最新保存的文件之外的其他文件
                for file in files:
                    # if file != latest_file:
                    os.remove(file)
                print("Old files deleted successfully.")
            else:
                print("No old files to delete.")
        except Exception as e:
            print("Error occurred while deleting old files:", e)

    def writer_img(self, path, frame):
        cv2.imwrite(path, frame)
        # print("----------------------完成--------------")

    def run(self):
        global rear_truck, front_img
        print(
            f"------------------ImageProcessWorker2---id:{self.camera_id},roi:{self.roi},file_path:{self.file_path}------------------执行")
        res_dict_lst = []
        rear_count = 0
        while True:
            if not self.queue.empty():
                print(f"--------------ImageProcessWorkerRear-----------------{self.queue.qsize()}")
                camera_id, frame = self.queue.get()

                frame = frame[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]

                # path = f"my_test_img/{self.result}/{time_str()}.jpg"
                # grabber = threading.Thread(target=self.writer_img, args=(path, frame))
                # grabber.start()

                # self.frame_counter += 1
                # # rec队列采集
                # if self.frame_counter == 5:
                #     self.rec_queue.put([self.camera_id, frame])
                #     self.frame_counter = 0

                start_time = time.time()
                self.my_container_detect.process_frame(frame)
                _, reuslt_dict = self.my_container_detect.get_result()
                end_time = time.time()
                print(f"------后摄像头识别时间: {end_time - start_time:.6f} 秒")


                if reuslt_dict:
                    res_dict_lst.append(reuslt_dict)
                    self.frame_counter += 1
                    if self.frame_counter >= 15:
                        self.dataQueued.emit(self.camera_id, reuslt_dict["img"])
                        rear_truck = "ok"

                else:
                    rear_count += 1
                    if rear_count >= 5:
                        rear_truck = "no"
                    self.frame_counter = 0

                print(f"--------------rear :rear_truck {rear_truck}")

                if self.my_container_detect.non_truck > 5 and self.my_container_detect.new_truck:
                    # 当连续10帧(约1s)没有集装箱面，且之前有卡车进入时，获取前一段时间面积最大帧
                    reuslt_dict, _ = self.my_container_detect.get_result()
                    print(reuslt_dict['area'])
                    # ocr队列采集
                    self.ocr_queue.put([(self.camera_id + 6), reuslt_dict['img']])
                    print("----------------------传入数据---ocr_queue----")

                    final_label, showImg = self.my_container_detect.door_label_vote(res_dict_lst)

                    # 界面刷新
                    self.dataQueued.emit(self.camera_id, reuslt_dict["img"])
                    rear_truck = "ok"
                    if front_img is not None and front_img.size > 0:
                        self.dataQueued.emit(0, front_img)
                        front_img = None

                    print("当连续10帧(约1s)没有集装箱面，且之前有卡车进入时，获取前一段时间面积最大帧", final_label)
                    if final_label == 'door':
                        # print(f"final_label == 'door': self.camera_id:{self.camera_id}")
                        self.ocr_queue.put([(self.camera_id + 10), None])
                        print("----------------------传入数据---ocr_queue----")
                    else:
                        self.ocr_queue.put([(self.camera_id + 20), None])

                    file_path = f'tmp/{self.camera_id + 3}.jpg'

                    cv2.imwrite(file_path, reuslt_dict["img"])
                    # cv2.imwrite(file_path, showImg)

                    print(f"ImageProcessWorker2: self.camera_id == 1")
                    self.rec_queue.put([5, None])
                    self.ocr_queue.put([5, None])
                    print("----------------------传入数据---ocr_queue----rec_queue")
                    logging.info('car away out ')
                    # !!! 获取最大面积图像后刷新是否有车的状态、刷新存下的结果
                    self.my_container_detect.new_truck = False
                    self.my_container_detect.max_area_dict.clear()
                    self.my_container_detect.res_dict.clear()
                    res_dict_lst.clear()


class SaveProcessWorker(QThread):
    def __init__(self, path, channel):
        super().__init__()
        self.path = path
        self.ch = channel
        self.image_number = '00'
        self.vehicle_count = "CAR0001"
        import socket
        import zmq
        self.ip_address = socket.gethostbyname(socket.gethostname())
        # ZeroMQ 上下文
        self.context = zmq.Context()
        # 创建发布者Socket，并绑定到指定端口
        self.publisher = self.context.socket(zmq.PUB)
        self.publisher.bind("tcp://*:5555")  # 本地绑定，使用端口5555

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
        print(output_folder)
        self.check_and_save(current_date, current_time, output_folder)

    def check_and_save(self, current_date, current_time, output_folder):
        global xiangti, xiangti_count
        position = ["left", "top", "right", "Front", "rear", ]
        li = []
        for i in range(5):
            # 图的名称
            filename = f"{self.ch}_{current_date}_{current_time}_{self.vehicle_count}_Transfer_{position[i]}_{self.image_number}.jpg"
            print(f"Generated filename: {filename}")
            # 每张图的最终路径
            output_path = os.path.join(output_folder, filename)
            print(f"Full output path: {output_path}")
            file_path = f'tmp/{i}.jpg'
            # os.rename(file_path,output_path)
            self.rename_file(file_path, output_path)
            li.append(output_path)

        if xiangti == "双箱":
            xiangti_count = 0

        self.rename_file("tmp/ocr.txt", os.path.join(output_folder, "ocr.txt"))
        ocr_path = os.path.join(output_folder, "ocr.txt")
        self.rename_file("tmp/rec.txt", os.path.join(output_folder, "rec.txt"))
        if os.path.exists('tmp/chepai.jpg'):
            self.rename_file("tmp/chepai.jpg", os.path.join(output_folder, "chepai.jpg"))

        try:
            if xiangti == "":
                time.sleep(3)
            with open(ocr_path, 'a', encoding='utf-8') as file:
                file.write(xiangti + '/n')

            xiangti = ''
        except Exception as e:
            # 在这里处理异常
            print("An error occurred:", e)
        # 获取当前时间
        current_time = datetime.now()
        # 格式化时间戳
        timestamp = current_time.strftime("%Y%m%d%H%M%S")
        # 构建数据字符串
        data = f"[C|{timestamp}|{self.ch}|{self.vehicle_count}|{li[3]}|{li[4]}|{li[0]}|{li[2]}|{li[1]}|{self.ip_address}|D]"
        # 发布数据
        self.publisher.send_string(data)
        # 打印已发送的数据
        print("Sent:", data)
        # tcp socket

    def check_files_existence(self, folder_path):
        # 要检查的文件列表
        files_to_check = ['0.jpg', '1.jpg', '2.jpg', '3.jpg', '4.jpg', 'ocr.txt', 'rec.txt']
        # 检查每个文件是否存在
        for file_name in files_to_check:
            file_path = os.path.join(folder_path, file_name)
            if not os.path.exists(file_path):
                return False
        return True

    def run(self):
        folder_path = 'tmp'
        while True:
            if self.check_files_existence(folder_path):
                # time.sleep(1)
                self.check()
                print("存档img...")
                logging.info('存档img... ')

            time.sleep(1)


class OcrThread(QThread):
    ocrMessage = Signal(int, list)

    def __init__(self, queue):
        QThread.__init__(self)
        config_dict = {
            "ocr_det_config": "./config/det/my_det_r50_db++_td_tr.yml",
            "ocr_rec_config": "./config/rec/my_en_PP-OCRv3_rec.yml"
        }
        self.ocr = det_ocr.OCR_process(config_dict)
        self.result = "ocr"
        self.queue = queue

    def writer_img(self, path, frame):
        cv2.imwrite(path, frame)
        # print("----------------------完成--------------")


    def run(self):
        global xiangti_count
        result_front = None
        result_rear = None
        result_left = None
        result_right = None
        door_in_front = ''
        door_in_rear = ''
        # door_in_left = ''
        # door_in_right = ''
        result_list_front = []
        result_list_rear = []
        result_list_left = []
        result_list_right = []
        while True:
            if not self.queue.empty():
                id, frame = self.queue.get()

                # path = f"my_test_img/{self.result}/{time_str()}.jpg"
                # grabber = threading.Thread(target=self.writer_img, args=(path, frame))
                # grabber.start()

                print(f"--------------OcrThread-----------------{self.queue.qsize()}")
                if id == 10:
                    door_in_front = "箱门朝前"
                    self.ocrMessage.emit(1, "箱门朝前")

                if id == 20:
                    door_in_front = "箱门朝后"

                if id == 11:
                    door_in_rear = "箱门朝后"
                    self.ocrMessage.emit(1, "箱门朝后")
                if id == 21:
                    door_in_rear = "箱门朝前"

                if id != 5:
                    start_time = time.time()
                    if id == 0:
                        result_left = self.ocr.process_imgs([frame], "left")
                        print(f"result_left----------------------{result_left}")
                    if id == 2:
                        result_right = self.ocr.process_imgs([frame], "right")
                        print(f"result_right----------------------{result_right}")
                    if id == 6:
                        frame_ues = cv2.GaussianBlur(frame, (5, 5), 0)
                        result_front = self.ocr.process_imgs([frame_ues, frame], "front")
                        print(f"result_front----------------------{result_front}")
                    if id == 7:
                        frame_ues = cv2.GaussianBlur(frame, (5, 5), 0)
                        result_rear = self.ocr.process_imgs([frame_ues, frame], "rear")
                        print(f"result_rear----------------------{result_rear}")
                    end_time = time.time()

                    print(f"------车箱号执行时间: {end_time-start_time:.6f} 秒")

                    if result_front is not None:
                        result1, result2 = result_front
                        if (result1 != '') & (result2 != ''):
                            result_list_front.append(result_front)
                        elif result1 != '' and result2 == '':
                            result_list_front.append((result1, ''))
                        elif result1 == '' and result2 != '':
                            result_list_front.append(('', result2))

                    if result_rear is not None:
                        result1, result2 = result_rear
                        if (result1 != '') & (result2 != ''):
                            result_list_rear.append(result_rear)
                        elif result1 != '' and result2 == '':
                            result_list_rear.append((result1, ''))
                        elif result1 == '' and result2 != '':
                            result_list_rear.append(('', result2))

                    if result_left is not None:
                        result1, result2 = result_left
                        if (result1 != '') & (result2 != ''):
                            result_list_left.append(result_left)
                        elif result1 != '' and result2 == '':
                            result_list_left.append((result1, ''))
                        elif result1 == '' and result2 != '':
                            result_list_left.append(('', result2))

                    if result_right is not None:
                        result1, result2 = result_right
                        if (result1 != '') & (result2 != ''):
                            result_list_right.append(result_right)
                        elif result1 != '' and result2 == '':
                            result_list_right.append((result1, ''))
                        elif result1 == '' and result2 != '':
                            result_list_right.append(('', result2))

                if id == 5:
                    file_path = 'tmp/ocr.txt'
                    start_time = time.time()
                    if result_list_front:
                        if xiangti_count < 1 or (xiangti == "双箱" and xiangti_count < 2):
                            ultra_result = det_ocr.vote2res(result_list_front)
                            print("front ultra ocr result:", ultra_result)
                            self.ocrMessage.emit(1, ultra_result)
                            # 写入结果到文件中
                            with open(file_path, 'a', encoding='utf-8') as file:
                                ultra_result = ultra_result[:-4] + ' ' + ultra_result[-4:]
                                file.write(ultra_result + '/n')
                                file.write(door_in_front + '/n')
                                file.write("-----------------------------" + '/n')
                                print("front ultra ocr result success:", ultra_result)
                            result_list_front = []
                            xiangti_count += 1
                            door_in_front = ''

                    if result_list_rear:
                        if xiangti_count < 1 or (xiangti == "双箱" and xiangti_count < 2):
                            ultra_result = det_ocr.vote2res(result_list_rear)
                            print("rear ultra ocr result:", ultra_result)
                            self.ocrMessage.emit(1, ultra_result)
                            # 写入结果到文件中
                            with open(file_path, 'a', encoding='utf-8') as file:
                                ultra_result = ultra_result[:-4] + ' ' + ultra_result[-4:]
                                file.write(ultra_result + '/n')
                                file.write(door_in_rear + '/n')
                                file.write("-----------------------------" + '/n')
                                print("rear ultra ocr result success:", ultra_result)
                            result_list_rear = []
                            xiangti_count += 1
                            door_in_rear = ''

                    if result_list_left:
                        if xiangti_count < 1 or (xiangti == "双箱" and xiangti_count < 2):
                            ultra_result = det_ocr.vote2res(result_list_left)
                            print("left ultra ocr result:", ultra_result)
                            self.ocrMessage.emit(1, ultra_result)
                            # 写入结果到文件中
                            with open(file_path, 'a', encoding='utf-8') as file:
                                ultra_result = ultra_result[:-4] + ' ' + ultra_result[-4:]
                                file.write(ultra_result + '/n')
                                if door_in_rear != '':
                                    file.write(door_in_rear + '/n')
                                    door_in_rear = ''
                                if door_in_front != '':
                                    file.write(door_in_rear + '/n')
                                    door_in_front = ''
                                file.write("-----------------------------" + '/n')
                                print("left ultra ocr result success:", ultra_result)
                            result_list_left = []
                            xiangti_count += 1

                    if result_list_right:
                        if xiangti_count < 1 or (xiangti == "双箱" and xiangti_count < 2):
                            ultra_result = det_ocr.vote2res(result_list_right)
                            print("right ultra ocr result:", ultra_result)
                            self.ocrMessage.emit(1, ultra_result)
                            # 写入结果到文件中
                            with open(file_path, 'a', encoding='utf-8') as file:
                                ultra_result = ultra_result[:-4] + ' ' + ultra_result[-4:]
                                file.write(ultra_result + '/n')
                                if door_in_rear != '':
                                    file.write(door_in_rear + '/n')
                                    door_in_rear = ''
                                if door_in_front != '':
                                    file.write(door_in_rear + '/n')
                                    door_in_front = ''
                                file.write("-----------------------------" + '/n')
                                print("right ultra ocr result success:", ultra_result)
                            result_list_right = []
                            xiangti_count += 1
                    end_time = time.time()
                    print(f"------车箱号写入文件以及前端展示执行时间: {end_time - start_time:.6f} 秒")
                    print("结果已保存到文件:", file_path)


class RecThread(QThread):
    recMessage = Signal(int, list)

    def __init__(self, queue):
        QThread.__init__(self)
        weights_dict = {
            "ocr_det_config": "./config_car/det/my_car_det_r50_db++_td_tr.yml",
            "ocr_rec_config": "./config_car/rec/my_rec_chinese_lite_train_v2.0.yml"
        }

        self.lp = det_ocr_car.OCR_process(weights_dict)
        # self.lp = yr.License_process(weights_dict)
        self.queue = queue
        self.result = "rec"
        self.cont_num = 1

    def writer_img(self, path, frame):
        cv2.imwrite(path, frame)
        # print("----------------------车牌照片写入完成--------------")

    def writer_img_two(self, path, frame):
        cv2.imwrite(path, frame)
        # print("----------------------车牌照片写入完成--------------")

    def writer_file(self,path,model,data,encoding=False):
        if encoding:
            with open(path, model, encoding=encoding) as file:
                file.write(data)
        else:
            with open(path, model) as fw:
                fw.write(str(self.cont_num) + "/t" + data + "/n")
                self.cont_num += 1


    def run(self):
        global front_img
        result_list = []
        save_frame = []
        while True:
            if not self.queue.empty():
                id, frame = self.queue.get()
                print(f"--------------RecThread-----------------{self.queue.qsize()}")

                # path = f"my_test_img/{self.result}/{time_str()}.jpg"
                # grabber = threading.Thread(target=self.writer_img, args=(path, frame))
                # grabber.start()
                if id != 5:
                    # path = f"my_test_img/shibie_chepai/{time_str()}.jpg"
                    # grabber = threading.Thread(target=self.writer_img_two, args=(path, frame))
                    # grabber.start()

                    # img = cv2.imread(r"C:/Users/Install/Desktop/2024_07_17_14_07_05.jpg")

                    start_time = time.time()
                    use_frame = cv2.GaussianBlur(frame, (5, 5), 0)
                    result = self.lp.process_imgs([use_frame])

                    if len(result) > 0:
                        print("---车牌识别结果------成功为:", result, "id 为:", id)
                        # path = f"my_test_img/chepai/{time_str()}.jpg"
                        # grabber = threading.Thread(target=self.writer_img_two, args=(path, frame))
                        # grabber.start()
                        result_list.append([result, frame])
                        save_frame.append(frame)

                    end_time = time.time()
                    print(f"------车牌识别时间: {end_time - start_time:.6f} 秒")

                if id == 5:

                    start_time = time.time()

                    ultra_result, save_frame_idx = det_ocr_car.get_finalResult(result_list)
                    if len(save_frame) > 0:
                        final_save = save_frame[save_frame_idx]
                        # cv2.imwrite('tmp/chepai.jpg', final_save)
                        chepai_data = threading.Thread(target=self.writer_img, args=('tmp/chepai.jpg', final_save))
                        chepai_data.start()
                        print("--------------tmp/chepai.jpg-----------------")

                    # with open("0611-001-reccar.txt", "a") as fw:
                    #     fw.write(str(self.cont_num) + "/t" + ultra_result + "/n")
                    #     self.cont_num += 1

                    reccar = threading.Thread(target=self.writer_file, args=('0611-001-reccar.txt', "a",ultra_result))
                    reccar.start()

                    self.recMessage.emit(1, ultra_result)

                    result_list = []
                    save_frame = []
                    print("ultra rec result:", ultra_result)

                    # 写入结果到文件中
                    rec_path = 'tmp/rec.txt'
                    # with open(file_path, 'w', encoding='utf-8') as file:
                    #     file.write(ultra_result)
                    tmp_rec = threading.Thread(target=self.writer_file, args=(rec_path, 'w', ultra_result,'utf-8'))
                    tmp_rec.start()

                    print("结果已保存到文件:", rec_path)

                    end_time = time.time()
                    print(f"------车牌保存时间: {end_time - start_time:.6f} 秒")



class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.labels1 = [self.ui.label, self.ui.label_2, self.ui.label_3]
        self.labels2 = [self.ui.label_4, self.ui.label_5]
        self.labels = [self.ui.label_7, self.ui.label_8, self.ui.label_9]

        with open('camera.json', 'r') as f:
            self.camera_config = json.load(f)

        self.ocr_queue = Queue()
        self.rec_queue = Queue()

        self.ocr_thread = OcrThread(self.ocr_queue)
        self.ocr_thread.ocrMessage.connect(self.update_ocr)

        self.rec_thread = RecThread(self.rec_queue)
        self.rec_thread.recMessage.connect(self.update_rec)

        self.path = "result"
        self.channel = "01"
        self.save_thread = SaveProcessWorker(self.path, self.channel)

        self.image_queues = [Queue() for _ in range(len(self.camera_config))]

        self.image_process_workers = [ImageProcessWorker(camera_info, queue, self.ocr_queue) for camera_info, queue in
                                      zip(list(self.camera_config.values())[:3], self.image_queues[:3])]

        for worker in self.image_process_workers:
            worker.image_processed.connect(self.handle_image_processed)
            worker.dataQueued.connect(self.update_label_1)
            worker.frameCaptured.connect(self.update_label)

        self.additional_image_workers_front = ImageProcessWorkerFront(
            list(self.camera_config.values())[3],
            self.image_queues[3],
            self.rec_queue,
            self.ocr_queue
        )

        self.additional_image_workers_front.image_processed.connect(self.handle_image_processed)
        self.additional_image_workers_front.dataQueued.connect(self.update_label_2)

        self.additional_image_workers_rear = ImageProcessWorkerRear(
            list(self.camera_config.values())[4],
            self.image_queues[4],
            self.rec_queue,
            self.ocr_queue
        )

        self.additional_image_workers_rear.image_processed.connect(self.handle_image_processed)
        self.additional_image_workers_rear.dataQueued.connect(self.update_label_2)

        self.capture_processes = [ImageCaptureProcess(camera_info, queue) for camera_info, queue in
                                  zip(self.camera_config.values(), self.image_queues)]
        # 单独设置进程名称
        for i, process in enumerate(self.capture_processes):
            process.name = f"CaptureProcess_{i}"

        self.start()

    def start(self):
        logging.info('Program start')
        os.makedirs('tmp', exist_ok=True)

        self.ocr_thread.start()
        self.rec_thread.start()
        self.save_thread.start()
        self.additional_image_workers_front.start()
        self.additional_image_workers_rear.start()

        for worker in self.image_process_workers:
            worker.start()

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
        self.ui.textEdit.insertPlainText(current_time + '/n')
        result_text = f"Ocr Result: {sorted_data} /n"
        self.ui.textEdit.insertPlainText(result_text)
        self.ui.textEdit.ensureCursorVisible()

    def update_rec(self, camera_id, result):
        current_time = QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")
        self.ui.textEdit.insertPlainText(current_time + '/n')
        result_text = f"Rec Result: {result} /n"
        self.ui.textEdit.insertPlainText(result_text)
        self.ui.textEdit.ensureCursorVisible()

    def handle_image_processed(self, camera_id, image):
        # 处理已处理的图像
        if camera_id == 20:
            result_text = f"Yl Result: 双箱 /n"
            self.ui.textEdit.insertPlainText(result_text)
            self.ui.textEdit.ensureCursorVisible()

        if camera_id == 21:
            result_text = f"Yl Result: 单箱 /n"
            self.ui.textEdit.insertPlainText(result_text)
            self.ui.textEdit.ensureCursorVisible()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    app = QApplication([])
    main_window = MainWindow()
    main_window.show()
    app.exec()
