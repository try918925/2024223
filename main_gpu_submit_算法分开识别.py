# -*- coding: utf-8 -*-
import json
import os
os.add_dll_directory(r"C:/opencv-4.9.0/build/install/x64/vc16/bin")
os.add_dll_directory(r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin")
import cv2
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
import ctypes

run_Flag = True
xiangti = ''
rear_truck = False

detector = YOLOv5Detector.from_config(my_config)


def time_str(fmt=None):
    if fmt is None:
        fmt = '%Y_%m_%d_%H_%M_%S_%f'
    return datetime.today().strftime(fmt)


class ImageCaptureProcess(Process):
    def __init__(self, camera_info, queue):
        super().__init__()
        self.camera_id = camera_info['id']
        self.file_path = camera_info['file_path']
        self.direction = camera_info['direction']
        self.roi = camera_info.get('roi', None)
        self.queue = queue
        self.dsize = (1000, 600)
        self.front_rear = ("front", "rear")
        self.left_right_top = ("right", "left", "top")
        self.roi_cols = (self.roi[2], self.roi[3])
        self.roi_rows = (self.roi[0], self.roi[1])

        # 设置透视变换矩阵
        trapezoid_r = np.array([[450, 637], [2120, 684], [456, 880], [2118, 811]], dtype=np.float32)
        rectangle_r = np.array([[200, 0], [1040, 0], [200, 300], [1040, 300]], dtype=np.float32)
        self.M_right = cv2.getPerspectiveTransform(trapezoid_r, rectangle_r)

        trapezoid_l = np.array([[392, 635], [2130, 560], [393, 795], [2126, 872]], dtype=np.float32)
        rectangle_l = np.array([[200, 0], [1040, 0], [200, 300], [1040, 300]], dtype=np.float32)
        self.M_left = cv2.getPerspectiveTransform(trapezoid_l, rectangle_l)

        # 畸变系数和相机内参
        self.distortion_coeffs = np.array(
            [-7.7041889e+00, 1.909734151e+00, -2.68401705e-03, 8.63032028e-02, -1.07008640e+02])
        self.camera_matrix = np.array([[9.73243587e+03, 0.00000000e+00, 1.22599216e+03],
                                       [0.00000000e+00, 9.58544217e+03, 7.26588013e+02],
                                       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    def run(self):
        self.counter = 0
        try:
            decoder = cv2.cudacodec.createVideoReader(self.file_path)
            count = 0
            start_time = time.time()  # 记录起始时间
            while True:
                ret, frame = decoder.nextFrame()
                if not ret or frame is None:
                    # 如果读取失败，重新初始化 decoder
                    decoder = cv2.cudacodec.createVideoReader(self.file_path)
                    print(f"{self.direction}:识别断流........")
                    continue

                # 前后方向的处理逻辑
                if self.direction in self.front_rear:
                    self.counter += 1
                    if self.counter % 2 == 0:
                        self.put_rear_front(frame)
                        self.counter = 0
                else:
                    self.put_left_right_top(frame)

                count += 1  # 计数每帧

                # 每秒统计读取的图片数量
                now_time = time.time()
                elapsed_time = now_time - start_time
                if elapsed_time >= 1:
                    # print(f"{self.direction}: 在 {elapsed_time:.2f} 秒内读取了 {count} 张图片")
                    count = 0  # 重置计数
                    start_time = time.time()  # 重新开始计时

        except Exception as error:
            print(f"ImageCaptureProcess---{self.direction}:图片读取有问题:{error}")

    def put_rear_front(self, frame):
        frame_crop = frame.colRange(self.roi_cols[0], self.roi_cols[1]).rowRange(self.roi_rows[0], self.roi_rows[1])
        frame_cpu = frame_crop.download()  # 裁剪后的图像下载到 CPU
        frame_cpu = frame_cpu[:, :, :3]  # 裁剪通道
        self.queue.put((self.camera_id, frame_cpu))

    def put_left_right_top(self, frame):
        frame_resized = cv2.cuda.resize(frame, self.dsize)  # 对原图进行 resize
        frame_crop = frame.colRange(self.roi_cols[0], self.roi_cols[1]).rowRange(self.roi_rows[0], self.roi_rows[1])
        frame_cpu_crop = frame_crop.download()  # 裁剪后的图像
        frame_cpu_resized = frame_resized.download()  # 调整大小后的图像
        frame_cpu_crop = frame_cpu_crop[:, :, :3]
        frame_cpu_resized = frame_cpu_resized[:, :, :3]
        self.queue.put((self.camera_id, frame_cpu_resized, frame_cpu_crop))

    # def crop_images(self, img, v):
    #     start_time = time.time()
    #     if v == 'right':
    #         result = cv2.warpPerspective(img, self.M_right, (2560, 1440))
    #         result = cv2.rotate(result, cv2.ROTATE_90_CLOCKWISE)
    #         result = result[130:1220, 940:1540]
    #         # print("crop_images(right):", time.time() - start_time)
    #         return result
    #     elif v == 'left':
    #         result = cv2.warpPerspective(img, self.M_left, (2560, 1440))
    #         result = cv2.rotate(result, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #         # print("crop_images(left):", time.time() - start_time)
    #         return result[1450:2540, 0:300]
    #     elif v == 'top':
    #         undistorted_img = cv2.undistort(img, self.camera_matrix, self.distortion_coeffs)
    #         result = cv2.rotate(undistorted_img, cv2.ROTATE_90_CLOCKWISE)
    #         # print("crop_images(top):", time.time() - start_time)
    #         return result


class ImageProcessWorker(QThread):
    frameCaptured = Signal(int, int)
    image_processed = Signal(int, object)
    dataQueued = Signal(int, object)

    def __init__(self, camera_info, result_queue, qu2, rec_queue):
        super().__init__()
        self.camera_id = camera_info['id']
        self.file_path = camera_info['file_path']
        self.direction = camera_info['direction']
        self.roi = camera_info.get('roi', None)
        self.result_queue = result_queue
        self.ocr_queue = qu2
        self.rec_queue = rec_queue
        self.true_threshold = 3
        self.initialize_inference()
        self.count = 0
        self.my_stiching = ts.stiching_img()

    def initialize_inference(self):
        cfg_dict = {
            "engine_path": "./tianJinGang.engine",
            "class": [0, 1]  # 0 有车，result：1没车
        }
        self.mobilenet_wrapper = trt_infer.MobilenetTRT(cfg_dict["engine_path"])
        # Warm-up 只进行一次以减少开销
        for _ in range(10):
            thread1 = trt_infer.warmUpThread(self.mobilenet_wrapper)
            thread1.start()
            thread1.join()

    def call_shuangxiang(self, frame):
        global xiangti
        img = np.ascontiguousarray(frame)
        img = cv2.resize(img, (4160, 1040))  # 保证数据和训练数据尺寸统一
        obj_list = detector.det([img])[0]
        xiangti = '双箱' if len(obj_list) == 2 else '单箱'
        self.image_processed.emit(20 if len(obj_list) == 2 else 21, None)
        return 2 if len(obj_list) == 2 else 1

    def process_frames(self, frames, id, thread_num, status):
        # print(f"{self.direction}, :拼图的图片数量为:{len(frames)}")
        # start_time = time.time()
        # print(self.direction, ":拼图开始的时间为:", start_time)

        stitching_method = {1: "top", 0: "left", 2: "right"}.get(id, "left")
        result = self.my_stiching.stiching(frames, stitching_method, thread_num, status)

        if result:
            if id == 2:
                threading.Thread(target=self.call_shuangxiang, args=(result,)).start()
            resized_image = cv2.resize(result, (1278, 344))
            self.dataQueued.emit(id, resized_image)
            cv2.imwrite(f'tmp/{id}.jpg', result)

    def run(self):
        self.frames_to_process = []
        self.car_in = False
        self.consecutive_true_count = 0
        self.consecutive_zero_count = 0
        self.thread_num = 0
        while run_Flag:
            if not self.result_queue.empty():
                # print(self.direction, ":队列的长度为：", self.result_queue.qsize())
                self.camera_id, frame, seg_frame = self.result_queue.get()
                self.count += 1
                if self.count % 2 == 0:
                    # start_time = time.time()
                    thread1 = trt_infer.inferThread(self.mobilenet_wrapper, [seg_frame])
                    thread1.start()
                    thread1.join()
                    result = thread1.get_result()
                    # print("车辆识别时间为:", time.time() - start_time)
                else:
                    result = "NO"

                if result == 1:
                    self.consecutive_zero_count = 0
                    self.consecutive_true_count += 1
                    if self.consecutive_true_count >= self.true_threshold:
                        self.consecutive_true_count = 0
                        # if len(self.frames_to_process) < 5:
                        #     self.frames_to_process = []
                        if self.car_in:
                            self.car_in = False
                            self.frameCaptured.emit(self.camera_id, 0)
                            threading.Thread(target=self.process_frames,
                                             args=(
                                                 self.frames_to_process, self.camera_id, self.thread_num, True)).start()
                            self.thread_num = 0

                            print("--------------开启线程拼图-----------------")

                            self.frames_to_process = []

                elif result == 0:
                    frame = cv2.resize(frame, (2560, 1440))
                    self.consecutive_true_count = 0
                    self.consecutive_zero_count += 1
                    self.frames_to_process.append(frame)
                    if len(self.frames_to_process) >= 50:
                        threading.Thread(target=self.process_frames,
                                         args=(self.frames_to_process, self.camera_id, self.thread_num, False)).start()
                        self.thread_num += 1
                        self.frames_to_process = []
                    if self.consecutive_zero_count > 3:
                        self.car_in = True
                        self.frameCaptured.emit(self.camera_id, 1)
                        self.image_processed.emit(self.camera_id, None)

                elif result == "NO" and self.car_in:
                    frame = cv2.resize(frame, (2560, 1440))
                    self.frames_to_process.append(frame)
                time.sleep(0.0002)


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
                for file in files:
                    os.remove(file)
                print("Old files deleted successfully.")
        except Exception as e:
            print("Error occurred while deleting old files:", e)

    def run(self):
        global rear_truck
        res_dict_lst = []
        rear_count = 0
        while run_Flag:
            try:
                if not self.queue.empty():
                    self.camera_id, frame = self.queue.get()
                    # print("ImageProcessWorker2", self.camera_id, self.queue.qsize(), "/n")
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
                        if self.camera_id == 1:
                            rear_count += 1
                        if rear_count >= 50:
                            if not rear_truck:
                                self.dataQueued.emit(self.camera_id, reuslt_dict['img'])
                            rear_count = 0
                    # print(f"{self.direction}的算法时间为:{time.time() - suanfa_time}")
                    if self.my_container_detect.non_truck > 5 and self.my_container_detect.new_truck:
                        # 当连续10帧(约1s)没有集装箱面，且之前有卡车进入时，获取前一段时间面积最大帧
                        # print('当连续18帧(约1s)没有集装箱面，且之前有卡车进入时，获取前一段时间面积最大帧')
                        # tuili_time = time.time()
                        reuslt_dict, _ = self.my_container_detect.get_result()
                        # ocr队列采集
                        self.ocr_queue.put(((self.camera_id + 6), reuslt_dict['img']))
                        final_label, _ = self.my_container_detect.door_label_vote(res_dict_lst)
                        # print(f"{self.direction}:集装箱面的推理时间为:{time.time() - tuili_time}")

                        # 界面刷新
                        self.dataQueued.emit(self.camera_id, reuslt_dict['img'])
                        if final_label == 'door':
                            self.ocr_queue.put(((self.camera_id + 10), None))
                        self.clear_writer_img(self.camera_id, reuslt_dict["img"])
                        if self.camera_id == 1:
                            rear_truck = True
                            self.rec_queue.put((5, None))
                            self.ocr_queue.put((5, None))
                            print('---------car away out----------- ')
                        else:
                            rear_truck = False
                        # !!! 获取最大面积图像后刷新是否有车的状态、刷新存下的结果
                        self.my_container_detect.new_truck = False
                        self.my_container_detect.max_area_dict.clear()
                        self.my_container_detect.res_dict.clear()
                        res_dict_lst.clear()
                    # print(f"{self.direction}的总体时间为:{time.time() - start_time}")
            except Exception as error:
                print(f"ImageProcessWorker2--{self.direction}:error:{error}")
            time.sleep(0.0002)

    def clear_writer_img(self, camera_id, frame):
        file_path = f'tmp/{camera_id + 3}.jpg'
        if camera_id == 0:
            self.delete_old_files()
        cv2.imwrite(file_path, frame)


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
            with open(ocr_path, 'a', encoding='utf-8') as file:
                file.write(xiangti + '/n')
            print("----执行完成----")

            xiangti = ''
        except Exception as e:
            # 在这里处理异常
            print("An error occurred:", e)

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


config_dict = {
    "ocr_det_config": "./config/det/my_det_r50_db++_td_tr.yml",
    "ocr_rec_config": "./config/rec/my_en_PP-OCRv3_rec.yml"
}
ocr = det_ocr.OCR_process(config_dict)


class OcrThread(QThread):
    ocrMessage = Signal(int, list)

    def __init__(self, queue):
        QThread.__init__(self)
        self.queue = queue

    def write_frame(self, path, frame):
        cv2.imwrite(path, frame)

    def run(self):
        ocr_data_dict = {"front": "", "rear": ""}
        result_list = []
        result = None
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
                    result = ocr.process_imgs([frame], "left")
                if id == 2:
                    result = ocr.process_imgs([frame], "right")
                if id == 6:
                    frame_ues = cv2.GaussianBlur(frame, (5, 5), 0)
                    result = ocr.process_imgs([frame_ues, frame], "front")
                if id == 7:
                    frame_ues = cv2.GaussianBlur(frame, (5, 5), 0)
                    result = ocr.process_imgs([frame_ues, frame], "rear")

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

            time.sleep(0.002)

    def write_ocr_data(self, ultra_result, ocr_data_dict):
        file_path = 'tmp/ocr.txt'
        # 写入结果到文件中
        print("准备写入数据orc:", ultra_result, ocr_data_dict)
        with open(file_path, 'w', encoding='utf-8') as file:
            ultra_result = ultra_result[:-4] + ' ' + ultra_result[-4:]
            file.write(ultra_result + '/n')
            if ocr_data_dict['front']:
                file.write(ocr_data_dict['front'] + '/n')
            if ocr_data_dict['rear']:
                file.write(ocr_data_dict['rear'] + '/n')
        print("结果已保存到文件:", 'tmp/ocr.txt')


weights_dict = {
    "ocr_det_config": "./config_car/det/my_car_det_r50_db++_td_tr.yml",
    "ocr_rec_config": "./config_car/rec/my_rec_chinese_lite_train_v2.0.yml"
}
lp = det_ocr_car.OCR_process(weights_dict)


class RecThread(QThread):
    recMessage = Signal(int, list)

    def __init__(self, queue):
        QThread.__init__(self)
        self.queue = queue
        self.cont_num = 1

    def run(self):
        result_list = []
        save_frame = []

        while run_Flag:
            id, frame = self.queue.get()
            if id != 5:
                use_frame = cv2.GaussianBlur(frame, (5, 5), 0)
                result = lp.process_imgs([use_frame])
                if len(result) > 0:
                    result_list.append([result, frame])
                    save_frame.append(frame)

            if id == 5:
                ultra_result, save_frame_idx = det_ocr_car.get_finalResult(result_list)
                if len(save_frame) > 0:
                    final_save = save_frame[save_frame_idx]
                    self.writer_img('tmp/chepai.jpg', final_save)
                self.recMessage.emit(1, ultra_result)
                result_list = []
                save_frame = []
                self.writer_rec_data(ultra_result)
            time.sleep(0.002)

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


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.labels1 = [self.ui.label, self.ui.label_2, self.ui.label_3]
        self.labels2 = [self.ui.label_4, self.ui.label_5]
        self.labels = [self.ui.label_7, self.ui.label_8, self.ui.label_9]

        with open('camera_model.json', 'r') as f:
            # with open('camera.json', 'r') as f:
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
        # self.result_queues = [Queue() for _ in range(len(self.camera_config))]

        self.image_process_workers = [ImageProcessWorker(camera_info, result_queues, self.ocr_queue, self.rec_queue) for
                                      camera_info, result_queues
                                      in
                                      zip(list(self.camera_config.values())[:3], self.original_queues[:3])]

        for worker in self.image_process_workers:
            worker.image_processed.connect(self.handle_image_processed)
            worker.dataQueued.connect(self.update_label_1)
            worker.frameCaptured.connect(self.update_label)

        # self.additional_image_workers = [ImageProcessWorker2(camera_info, queue, self.rec_queue, self.ocr_queue) for
        #                                  camera_info, queue in
        #                                  zip(list(self.camera_config.values())[3:], self.original_queues[3:])]

        # for worker in self.additional_image_workers:
        #     worker.image_processed.connect(self.handle_image_processed)
        #     worker.dataQueued.connect(self.update_label_2)

        # self.recognize_processes = [ImageProcessRecognize(original_queues, result_queues) for
        #                             original_queues, result_queues in
        #                             zip(self.original_queues[:3], self.result_queues[:3])]

        self.capture_processes = [ImageCaptureProcess(camera_info, original_queues) for camera_info, original_queues in
                                  zip(self.camera_config.values(), self.original_queues)]

        self.start()

    def start(self):
        os.makedirs('tmp', exist_ok=True)

        self.ocr_thread.start()
        self.rec_thread.start()
        self.save_thread.start()

        for worker in self.image_process_workers:
            worker.start()

        # for worker in self.additional_image_workers:
        #     worker.start()

        # for recognize in self.recognize_processes:
        #     recognize.start()

        time.sleep(8)

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
    freeze_support()
    app = QApplication([])
    main_window = MainWindow()
    main_window.show()
    app.exec()
