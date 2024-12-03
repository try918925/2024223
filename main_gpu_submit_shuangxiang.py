# -*- coding: utf-8 -*-
import json
import os
os.add_dll_directory(r"C:/opencv_build_data/opencv-4.9.0/build/install/x64/vc16/bin")
os.add_dll_directory(r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin")
import cv2
import ctypes
import numpy as np
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *
from uuu import Ui_MainWindow
# import threading
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from multiprocessing import Queue, Process
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

# from LOGGER import logger

run_Flag = True
xiangti = ''


class ImageCaptureProcess(Process):
    def __init__(self, camera_info, queue):
        super().__init__()
        self.camera_id = camera_info['id']
        self.file_path = camera_info['file_path']
        self.direction = camera_info['direction']
        self.roi = camera_info.get('roi', None)
        self.queue = queue
        self.front_rear = ("front", "rear")
        self.left_right_top = ("right", "left", "top")
        # self.dsize = (900, 900)
        self.dsize = (1066, 600)
        self.roi_cols = (self.roi[2], self.roi[3])
        self.roi_rows = (self.roi[0], self.roi[1])

    def run(self):
        try:
            decoder = cv2.cudacodec.createVideoReader(self.file_path)
            while True:
                # start_time = time.time()
                ret, frame = decoder.nextFrame()
                if not ret or frame is None:
                    decoder = cv2.cudacodec.createVideoReader(self.file_path)
                    print(f"{self.direction}:图片断流")
                    continue
                self.transfer_frame_gpu(frame)
                # self.transfer_frame_one(frame)
                # print(f"{self.direction}, 传入图片数量时间为:, {time.time() - start_time}")
        except Exception as e:
            print(f"ImageCaptureProcess---{self.direction}:图片读取有问题")

    def transfer_frame_gpu(self, frame):
        try:
            if self.direction in self.front_rear:
                frame_crop = frame.colRange(self.roi_cols[0], self.roi_cols[1]).rowRange(self.roi_rows[0], self.roi_rows[1])
                frame_cpu = frame_crop.download()  # 裁剪后的图像下载到 CPU
                frame_cpu = frame_cpu[:, :, :3]  # 裁剪通道
                self.queue.put((self.camera_id, frame_cpu))
            elif self.direction in self.left_right_top:
                frame_resized = cv2.cuda.resize(frame, self.dsize)  # 对原图进行 resize
                frame_crop = frame.colRange(self.roi_cols[0], self.roi_cols[1]).rowRange(self.roi_rows[0], self.roi_rows[1])
                frame_cpu_crop = frame_crop.download()  # 裁剪后的图像
                frame_cpu_resized = frame_resized.download()  # 调整大小后的图像
                frame_cpu_crop = frame_cpu_crop[:, :, :3]
                frame_cpu_resized = frame_cpu_resized[:, :, :3]
                self.queue.put((self.camera_id, frame_cpu_resized, frame_cpu_crop))
        except Exception as error:
            print(f"transfer_frame_gpu错误:{error}")

    def transfer_frame_one(self, frame):
        try:
            frame_cpu = frame.download()  # 裁剪后的图像下载到 CPU
            # print(frame_cpu.shape)
            frame_cpu = frame_cpu[:, :, :3]  # 裁剪通道
            self.queue.put((self.camera_id, frame_cpu))
        except Exception as error:
            print(self.direction, "错误:", error)


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
        self.consecutive_true_count = 0
        self.true_threshold = 3
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.detector = YOLOv5Detector.from_config(my_config)

    def call_shuangxiang(self, frame):

        global xiangti
        img = np.ascontiguousarray(frame)
        # 保证数据和训练数据尺寸统一
        img = cv2.resize(img, (4160, 1040))
        # tik = time.time()
        obj_list = self.detector.det([img])[0]
        # print(obj_list, time.time() - tik)
        # print(len(obj_list))
        if len(obj_list) == 2:
            # self.image_processed.emit(20, None)
            self.executor.submit(self.image_processed_emit, 20, None)
            xiangti = '双箱'
            return 2
        else:
            # self.image_processed.emit(21, None)
            self.executor.submit(self.image_processed_emit, 21, None)
            xiangti = '单箱'
            return 1

    def process_frames(self, frames, id):
        print(f"{self.direction}, :拼图的图片数量为:{len(frames)}", )
        my_stiching = ts.stiching_img()
        if id == 1:
            result = my_stiching.stiching(frames, "top")
        if id == 0:
            result = my_stiching.stiching(frames, "left")
        if id == 2:
            result = my_stiching.stiching(frames, "right")
            self.executor.submit(self.call_shuangxiang, result)

        self.executor.submit(self.writer_img, id, result)

        self.executor.submit(self.dataQueued_emit, id, result)

    def run(self):
        self.frames_to_process = []
        self.car_in = False
        consecutive_zero_count = 0
        self.count = 0
        while run_Flag:
            if not self.result_queue.empty():
                # start_time = time.time()
                self.camera_id, result, frame = self.result_queue.get()
                if result == 1:
                    consecutive_zero_count = 0
                    self.consecutive_true_count += 1
                    if self.consecutive_true_count >= self.true_threshold:
                        # 连续无车
                        self.consecutive_true_count = 0
                        if len(self.frames_to_process) < 3:
                            self.frames_to_process = []
                        # 标志位，代表车辆是否有进入记录
                        if self.car_in:
                            # consecutive_zero_count = 0
                            self.car_in = False
                            # self.frameCaptured.emit(self.camera_id, 0)
                            self.executor.submit(self.frameCaptured_emit, self.camera_id, 0)

                            # if self.camera_id != 1:
                            #     self.ocr_queue.put([5, None])

                            self.executor.submit(self.process_frames, self.frames_to_process.copy(), self.camera_id)
                            self.frames_to_process = []

                if result == 0:
                    frame = cv2.resize(frame, (2560, 1440))
                    self.consecutive_true_count = 0
                    consecutive_zero_count += 1
                    self.count + 1
                    self.frames_to_process.append(frame)
                    # ocr队列
                    if (consecutive_zero_count) % 2 == 0:
                        # # top不采集
                        # if self.camera_id != 1:
                        self.ocr_queue.put((self.camera_id, frame))
                    # 阈值
                    if consecutive_zero_count > 3:
                        self.car_in = True
                        # self.frameCaptured.emit(self.camera_id, 1)
                        self.executor.submit(self.frameCaptured_emit, self.camera_id, 1)
                        # self.image_processed.emit(self.camera_id, None)
                        self.executor.submit(self.image_processed_emit, self.camera_id, None)

                # print(f"{self.direction}相机:处理照片结果时间:", time.time() - start_time, "\n")

    def frameCaptured_emit(self, camera_id, data):
        self.frameCaptured.emit(camera_id, data)

    def image_processed_emit(self, camera_id, data):
        self.image_processed.emit(camera_id, data)

    def dataQueued_emit(self, id, result):
        resized_image = cv2.resize(result, (1278, 344))
        self.dataQueued.emit(id, resized_image)

    def writer_img(self, id, frame):
        file_path = f'tmp/{id}.jpg'
        cv2.imwrite(file_path, frame)


class ImageProcessRecognize(Process):
    def __init__(self, original_queue, result_queue):
        super().__init__()
        self.original_queue = original_queue
        self.result_queue = result_queue

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
        while run_Flag:
            if not self.original_queue.empty():
                # start_time = time.time()
                camera_id, frame, seg_frame = self.original_queue.get()
                thread1 = self.trt_infer.inferThread(self.mobilenet_wrapper, [seg_frame])
                thread1.start()
                thread1.join()
                result = thread1.get_result()
                # print(f"{camera_id}相机:识别算法时间:", time.time() - start_time, "\n")
                self.result_queue.put((camera_id, result, frame))
                # print(f"{camera_id}相机:完成算法结果时间:", time.time() - start_time, "\n")


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
        self.initialize_inference()
        self.executor = ThreadPoolExecutor(max_workers=3)

    def initialize_inference(self):
        PLUGIN_LIBRARY = "./myplugins.dll"
        engine_file_path = "truck_old.engine"
        ctypes.CDLL(PLUGIN_LIBRARY)
        self.csd_detector = cont_trt_infer.CSD_Detector(engine_file_path)  # 初始化detector
        self.my_container_detect = cont_trt_infer.container_detect(self.csd_detector)

    def delete_old_files(self):
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
        except Exception as e:
            print("Error occurred while deleting old files:", e)

    def run(self):
        res_dict_lst = []
        rear_count = 0
        while run_Flag:
            try:
                if not self.queue.empty():
                    # start_time = time.time()
                    camera_id, frame = self.queue.get()
                    self.frame_counter += 1
                    # rec队列采集 front==0
                    if self.camera_id == 0:
                        if self.frame_counter == 3:
                            self.rec_queue.put((self.camera_id, frame))
                            self.frame_counter = 0
                    suanfa_time = time.time()
                    self.my_container_detect.process_frame(frame)
                    _, reuslt_dict = self.my_container_detect.get_result()
                    if reuslt_dict:
                        res_dict_lst.append(reuslt_dict)
                        if camera_id == 1:
                            rear_count += 1
                        if rear_count >= 100:
                            self.executor.submit(self.interface_refresh, self.camera_id, reuslt_dict['img'])
                            rear_count = 0

                    # print(f"{self.direction}的算法时间为:{time.time() - suanfa_time}")

                    if self.my_container_detect.non_truck > 12 and self.my_container_detect.new_truck:
                        # 当连续10帧(约1s)没有集装箱面，且之前有卡车进入时，获取前一段时间面积最大帧
                        # print('当连续18帧(约1s)没有集装箱面，且之前有卡车进入时，获取前一段时间面积最大帧')

                        # tuili_time = time.time()
                        reuslt_dict, _ = self.my_container_detect.get_result()
                        # ocr队列采集
                        self.ocr_queue.put(((self.camera_id + 6), reuslt_dict['img']))
                        final_label, _ = self.my_container_detect.door_label_vote(res_dict_lst)
                        # print(f"{self.direction}:集装箱面的推理时间为:{time.time() - tuili_time}")

                        # 界面刷新
                        if self.camera_id == 0:
                            self.executor.submit(self.interface_refresh, self.camera_id, reuslt_dict['img'])

                        if final_label == 'door':
                            self.ocr_queue.put(((self.camera_id + 10), None))

                        self.executor.submit(self.clear_writer_img, self.camera_id, reuslt_dict["img"])

                        if self.camera_id == 1:
                            self.rec_queue.put((5, None))
                            self.ocr_queue.put((5, None))
                            print('---------car away out----------- ')
                        # !!! 获取最大面积图像后刷新是否有车的状态、刷新存下的结果
                        self.my_container_detect.new_truck = False
                        self.my_container_detect.max_area_dict.clear()
                        self.my_container_detect.res_dict.clear()
                        res_dict_lst.clear()
                    # print(f"{self.direction}的总体时间为:{time.time() - start_time}")
            except Exception as error:
                print(f"ImageProcessWorker2--{self.direction}:error:{error}")

    def interface_refresh(self, camera_id, image):
        self.dataQueued.emit(camera_id, image)

    def clear_writer_img(self, camera_id, frame):
        file_path = f'tmp/{camera_id + 3}.jpg'
        if camera_id == 0:
            self.delete_old_files()
        cv2.imwrite(file_path, frame)


def write_ocr_data(ocr_path, xiangti):
    ultra_result_one = ""
    ultra_result_two = ""

    with open(ocr_path, 'r', encoding='utf-8') as f:
        data_data = json.load(f)

    print("data_data:", data_data)

    start_time = time.time()

    if xiangti == "双箱":
        similar_data, dissimilar_data = returns_two_similar_data(data_data["ocr_data"])
        ultra_result_one = det_ocr.vote2res(similar_data)
        ultra_result_two = det_ocr.vote2res(dissimilar_data)
    elif xiangti == "单箱":
        ultra_result_one = det_ocr.vote2res(data_data["ocr_data"])

    print("ultra_result_one:",ultra_result_one,"ultra_result_two:",ultra_result_two)

    if ultra_result_one:
        ultra_result_one = f"{ultra_result_one[:-4]} {ultra_result_one[-4:]}"
    if ultra_result_two:
        ultra_result_two = f"{ultra_result_two[:-4]} {ultra_result_two[-4:]}"

    print(f"{xiangti}: 消耗时间: {time.time() - start_time}")

    with open(ocr_path, 'w', encoding='utf-8') as file:
        if ultra_result_one:
            file.write(ultra_result_one + '\n')
        if ultra_result_two:
            file.write(ultra_result_two + '\n')
        file.write(f"{data_data['front']}\n{data_data['rear']}\n{xiangti}")


def similarity_ratio(str1, str2):
    arr1 = np.array(list(str1))
    arr2 = np.array(list(str2))
    same_count = np.sum(arr1 == arr2)
    return same_count / len(str1)


def returns_two_similar_data(data, threshold=0.6):
    SIMILARITY_THRESHOLD = threshold
    if not data:
        return [], []
    similar_data = data[:1]
    dissimilar_data = []
    for item in data[1:]:
        added_to_similar = False
        for similar_item in similar_data:
            if similarity_ratio(item[0], similar_item[0]) >= SIMILARITY_THRESHOLD:
                similar_data.append(item)
                added_to_similar = True
                break
        if not added_to_similar:
            dissimilar_data.append(item)
    return similar_data, dissimilar_data


class SaveProcessWorker(QThread):
    def __init__(self, path, channel):
        super().__init__()
        self.path = path
        self.ch = channel
        self.image_number = '00'
        self.vehicle_count = "CAR0001"
        self.executor = ThreadPoolExecutor(max_workers=2)
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
        # print(output_folder)
        self.check_and_save(current_date, current_time, output_folder)

    def check_and_save(self, current_date, current_time, output_folder):
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

        # os.rename("tmp/ocr.txt", os.path.join(output_folder, "ocr.txt"))
        self.rename_file("tmp/ocr.txt", os.path.join(output_folder, "ocr.txt"))
        ocr_path = os.path.join(output_folder, "ocr.txt")
        # os.rename("tmp/rec.txt",os.path.join(output_folder, "rec.txt"))
        self.rename_file("tmp/rec.txt", os.path.join(output_folder, "rec.txt"))
        if os.path.exists('tmp/chepai.jpg'):
            self.rename_file("tmp/chepai.jpg", os.path.join(output_folder, "chepai.jpg"))

        try:
            global xiangti
            if xiangti == "":
                time.sleep(3)
            process = Process(target=write_ocr_data, args=(ocr_path, xiangti))
            process.start()
            print("----执行完成----")

            xiangti = ''
        except Exception as e:
            # 在这里处理异常
            print("An error occurred:", e)

        self.executor.submit(self.send_zmq_string, li)

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
        while run_Flag:
            if self.check_files_existence(folder_path):
                time.sleep(1)
                self.check()
                print('存档img... ')
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
        self.queue = queue
        self.executor = ThreadPoolExecutor(max_workers=4)

    def run(self):
        result = None
        ocr_data_dict = {"front": "", "rear": ""}
        result_list = []
        while run_Flag:
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
                print("一次性识别:", result_list)
                ocr_data_dict.update({"ocr_data": copy.deepcopy(result_list)})
                print("箱号识别数据:", ocr_data_dict)
                self.executor.submit(self.write_ocr_data, copy.deepcopy(ocr_data_dict))

                result_list.clear()
                ocr_data_dict = {"front": "", "rear": ""}

    def write_ocr_data(self, ocr_data_dict):
        file_path = 'tmp/ocr.txt'
        # 写入结果到文件中
        print("准备写入数据orc:", ocr_data_dict)
        with open(file_path, "w") as file:
            json.dump(ocr_data_dict, file)
        print("结果已保存到文件:", 'tmp/ocr.txt')

    def ocrMessage_emit(self, id, data):
        self.ocrMessage.emit(id, data)


class RecThread(QThread):
    recMessage = Signal(int, list)

    def __init__(self, queue):
        QThread.__init__(self)
        weights_dict = {
            "ocr_det_config": "./config_car/det/my_car_det_r50_db++_td_tr.yml",
            "ocr_rec_config": "./config_car/rec/my_rec_chinese_lite_train_v2.0.yml"
        }
        self.lp = det_ocr_car.OCR_process(weights_dict)
        self.queue = queue
        self.cont_num = 1
        self.executor = ThreadPoolExecutor(max_workers=4)

    def run(self):
        result_list = []
        save_frame = []

        while run_Flag:
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
                    # cv2.imwrite('tmp/chepai.jpg', final_save)
                    self.executor.submit(self.writer_img, 'tmp/chepai.jpg', final_save)

                # with open("0611-001-reccar.txt", "a") as fw:
                #     fw.write(str(self.cont_num) + "/t" + ultra_result + "/n")
                #     self.cont_num += 1
                self.executor.submit(self.record_reccar, ultra_result)

                # self.recMessage.emit(1, ultra_result)
                self.executor.submit(self.recMessage_emit, 1, ultra_result)

                result_list = []
                save_frame = []

                # 写入结果到文件中
                # file_path = 'tmp/rec.txt'
                # with open(file_path, 'w', encoding='utf-8') as file:
                #     file.write(ultra_result)
                self.executor.submit(self.writer_rec_data, ultra_result)

                # print("结果已保存到文件:file_path")

    def record_reccar(self, ultra_result):
        with open("0611-001-reccar.txt", "a") as fw:
            fw.write(str(self.cont_num) + "/t" + ultra_result + "/n")
            self.cont_num += 1

    def writer_rec_data(self, ultra_result):
        file_path = 'tmp/rec.txt'
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(ultra_result)

    def writer_img(self, file_path, frame):
        cv2.imwrite(file_path, frame)

    def recMessage_emit(self, id, data):
        self.recMessage.emit(id, data)


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.labels1 = [self.ui.label, self.ui.label_2, self.ui.label_3]
        self.labels2 = [self.ui.label_4, self.ui.label_5]
        self.labels = [self.ui.label_7, self.ui.label_8, self.ui.label_9]

        # with open('camera_model.json', 'r') as f:
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

        self.original_queues = [Queue() for _ in range(len(self.camera_config))]
        self.result_queues = [Queue() for _ in range(len(self.camera_config))]

        self.image_process_workers = [ImageProcessWorker(camera_info, result_queues, self.ocr_queue) for camera_info, result_queues
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
        # logging.info('Program start')
        os.makedirs('tmp', exist_ok=True)

        self.ocr_thread.start()
        self.rec_thread.start()
        self.save_thread.start()

        for worker in self.image_process_workers:
            worker.start()

        for worker in self.additional_image_workers:
            worker.start()

        for recognize in self.recognize_processes:
            recognize.start()

        time.sleep(5)

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
