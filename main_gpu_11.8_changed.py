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
import threading
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from multiprocessing import Queue, Process
import test_infer as trt_infer
import infer_det_rec as det_ocr
import infer_det_rec_car as det_ocr_car
import stichImg as ts
from algorithms.detector import YOLOv5Detector
# import Container_det as cont_trt_infer
import Container_det_trt_yolov5 as cont_trt_infer
import time
from datetime import datetime

# import logging

# 配置日志记录器
# logging.basicConfig(filename='log.txt', filemode='a', encoding='utf-8', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
run_Flag = True
xiangti = ''


def time_str(fmt=None):
    if fmt is None:
        fmt = '%Y_%m_%d_%H_%M_%S'
    return datetime.today().strftime(fmt)


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
        # self.dsize = (900, 400)
        self.roi_cols = (self.roi[2], self.roi[3])
        self.roi_rows = (self.roi[0], self.roi[1])
        self.executor = None

    def run(self):
        # self.executor = ThreadPoolExecutor(max_workers=30)
        frame_counter = 0
        count = 0
        try:
            decoder = cv2.cudacodec.createVideoReader(self.file_path)
            start_time = time.time()
            while True:
                # start_time = time.time()
                ret, frame = decoder.nextFrame()
                if not ret or frame is None:
                    decoder = cv2.cudacodec.createVideoReader(self.file_path)
                    print(self.direction, "图片读取有问题")
                    continue
                frame = frame.download()
                frame_counter += 1
                frame = frame[:, :, :3]
                # if frame_counter % 2 == 1:
                # self.executor.submit(self.transfer_frame_gpu, frame)
                self.queue.put((self.camera_id, frame))
                count += 1
                time_value = time.time() - start_time
                if time_value >= 1.0:
                    # 每秒打印传入的图像数量
                    print(self.direction, "传入图片数量为:", count, "在时间:", time_value, "\n")
                    # 重置计数器和开始时间
                    start_time = time.time()
                    count = 0


        except Exception as e:
            print("Error:", e)

    def transfer_frame(self, frame):
        if self.direction in self.front_rear:
            frame = frame[self.roi_rows[0]:self.roi_rows[1], self.roi_cols[0]:self.roi_cols[1]]
            self.queue.put((self.camera_id, frame))
        elif self.direction in self.left_right_top:
            frame_crop = frame[self.roi_rows[0]:self.roi_rows[1], self.roi_cols[0]:self.roi_cols[1]]
            frame_dsize = cv2.resize(frame, self.dsize)
            self.queue.put((self.camera_id, frame_dsize, frame_crop))

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
            # print(self.direction, "传入完成")
        except Exception as error:
            print(self.direction, "错误:", error)


class ImageProcessWorker(QThread):
    frameCaptured = Signal(int, int)
    image_processed = Signal(int, object)
    dataQueued = Signal(int, object)

    def __init__(self, camera_info, qu1, qu2):
        super().__init__()
        self.camera_id = camera_info['id']
        self.file_path = camera_info['file_path']
        self.direction = camera_info['direction']
        self.roi = camera_info.get('roi', None)
        self.queue = qu1
        self.ocr_queue = qu2
        self.executor = ThreadPoolExecutor(max_workers=15)
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

    def writer_img(self, path, frame):
        cv2.imwrite(path, frame)

    def process_frames(self, frames, id):
        print(self.direction, ":拼图的图片数量为:", len(frames))
        my_stiching = ts.stiching_img()
        if id == 1:
            result = my_stiching.stiching(frames, "top")
        if id == 0:
            result = my_stiching.stiching(frames, "left")
        if id == 2:
            result = my_stiching.stiching(frames, "right")
            thread = threading.Thread(target=self.call_shuangxiang, args=(result,))
            thread.start()
            # self.executor.submit(self.call_shuangxiang, result)
            # 启动线程

        file_path = f'tmp/{id}.jpg'
        # 保存图片
        cv2.imwrite(file_path, result)
        resized_image = cv2.resize(result, (1278, 344))
        self.dataQueued.emit(id, resized_image)

    def run(self):
        self.frames_to_process = []
        self.car_in = False
        consecutive_zero_count = 0
        while run_Flag:
            if not self.queue.empty():
                start_time = time.time()
                # camera_id, frame, seg_frame = self.queue.get()
                camera_id, frame = self.queue.get()
                seg_frame = frame[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
                img_list = []
                img_list.append(seg_frame)
                thread1 = trt_infer.inferThread(self.mobilenet_wrapper, img_list)
                thread1.start()
                thread1.join()
                print(f"{self.direction}相机:识别算法时间:", time.time() - start_time, "\n")
                del img_list
                result = thread1.get_result()
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
                            print("车辆驶离")
                            self.car_in = False
                            self.frameCaptured.emit(self.camera_id, 0)
                            # if self.camera_id != 1:
                            #     self.ocr_queue.put([5, None])
                            threading.Thread(target=self.process_frames, args=(self.frames_to_process.copy(), self.camera_id)).start()
                            self.frames_to_process = []

                if result == 0:
                    # original_frame = cv2.resize(frame, (2560, 1440))
                    self.consecutive_true_count = 0
                    consecutive_zero_count += 1
                    self.frames_to_process.append(frame)
                    # ocr队列
                    # if (consecutive_zero_count) % 2 == 0:
                    #     # top不采集
                    #     if self.camera_id != 1:
                    #         self.ocr_queue.put([self.camera_id, frame])
                    # 阈值
                    if consecutive_zero_count > 3:
                        self.car_in = True
                        self.frameCaptured.emit(self.camera_id, 1)
                        # self.image_processed.emit(camera_id, original_frame)
                        self.image_processed.emit(camera_id, None)

                print(f"{self.direction}相机:ImageProcessWorker:", time.time() - start_time, "\n")


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

        # from configs import config_5s_trt as door_config
        # self.csd_detector = YOLOv5Detector.from_config(door_config)

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

    def run(self):
        res_dict_lst = []
        count_ocr = 0
        while run_Flag:
            try:
                if not self.queue.empty():
                    start_time = time.time()
                    camera_id, frame = self.queue.get()
                    frame = frame[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
                    self.frame_counter += 1
                    # rec队列采集
                    if self.camera_id == 0:
                        if self.frame_counter == 3:
                            self.rec_queue.put([self.camera_id, frame])
                            self.frame_counter = 0

                    # start_time = time.time()
                    self.my_container_detect.process_frame(frame)
                    _, reuslt_dict = self.my_container_detect.get_result()
                    print(f"{self.direction}相机:识别算法时间:", time.time() - start_time, "\n")
                    if len(reuslt_dict) > 0:
                        res_dict_lst.append(reuslt_dict)
                        # count_ocr += 1
                        # if count_ocr == 3:
                        #     self.ocr_queue.put([(self.camera_id + 6),reuslt_dict['img']])
                        #     count_ocr = 0

                    if self.my_container_detect.non_truck > 14 and self.my_container_detect.new_truck:
                        # 当连续10帧(约1s)没有集装箱面，且之前有卡车进入时，获取前一段时间面积最大帧
                        reuslt_dict, _ = self.my_container_detect.get_result()
                        # print(reuslt_dict['area'])
                        # ocr队列采集
                        self.ocr_queue.put([(self.camera_id + 6), reuslt_dict['img']])

                        final_label, showImg = self.my_container_detect.door_label_vote(res_dict_lst)

                        # 界面刷新
                        # self.dataQueued.emit(self.camera_id, reuslt_dict["img"])
                        self.dataQueued.emit(self.camera_id, reuslt_dict['img'])
                        # 朝向
                        # print(reuslt_dict["label"]) # 输出结果为'door','nodoor'
                        # if reuslt_dict["label"] == 'door':
                        #     self.ocr_queue.put([(self.camera_id + 10),None])
                        print(final_label)
                        if final_label == 'door':
                            self.ocr_queue.put([(self.camera_id + 10), None])

                        file_path = f'tmp/{self.camera_id + 3}.jpg'
                        if self.camera_id == 0:
                            self.delete_old_files()
                            # logging.info('New car move in ')

                        cv2.imwrite(file_path, reuslt_dict["img"])
                        # cv2.imwrite(file_path, showImg)

                        if self.camera_id == 1:
                            self.rec_queue.put([5, None])
                            self.ocr_queue.put([5, None])
                            # logging.info('car away out ')
                        # !!! 获取最大面积图像后刷新是否有车的状态、刷新存下的结果
                        self.my_container_detect.new_truck = False
                        self.my_container_detect.max_area_dict.clear()
                        self.my_container_detect.res_dict.clear()
                        res_dict_lst.clear()
                    print(f"{self.direction}相机:ImageProcessWorker2:", time.time() - start_time, "\n")
            except Exception as error:
                print("ImageProcessWorker2:", error)


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
        while run_Flag:
            if self.check_files_existence(folder_path):
                time.sleep(1)
                self.check()
                print("存档img...")
                # logging.info('存档img... ')

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

    def run(self):
        result = None
        door_in = ''
        result_list = []
        while run_Flag:
            id, frame = self.queue.get()
            if id == 10:
                door_in = "箱门朝前"
                self.ocrMessage.emit(1, "箱门朝前")
            if id == 11:
                door_in = "箱门朝后"
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
                ultra_result = det_ocr.vote2res(result_list)
                result_list = []
                print("ultra ocr result:", ultra_result)
                self.ocrMessage.emit(1, ultra_result)
                file_path = 'tmp/ocr.txt'
                # 写入结果到文件中
                with open(file_path, 'w', encoding='utf-8') as file:
                    ultra_result = ultra_result[:-4] + ' ' + ultra_result[-4:]
                    file.write(ultra_result + '/n')
                    file.write(door_in + '/n')

                door_in = ''
                print("结果已保存到文件:", file_path)


class RecThread(QThread):
    recMessage = Signal(int, list)

    def __init__(self, queue):
        QThread.__init__(self)
        weights_dict = {
            "ocr_det_config": "./config_car/det/my_car_det_r50_db++_td_tr.yml",
            "ocr_rec_config": "./config_car/rec/my_rec_chinese_lite_train_v2.0.yml"
        }

        # weights_dict = {
        #     "yolo_weights": "weights/plate_detect.pt",
        #     "rec_weights": "weights/plate_rec_color.pth"
        # }
        self.lp = det_ocr_car.OCR_process(weights_dict)
        # self.lp = yr.License_process(weights_dict)
        self.queue = queue

    def run(self):
        result_list = []
        save_frame = []
        cont_num = 1
        # thres_lst = []
        while run_Flag:
            id, frame = self.queue.get()
            if id != 5:
                # use_frame = frame[500:1440, 600:1700]
                use_frame = cv2.GaussianBlur(frame, (5, 5), 0)
                result = self.lp.process_imgs([use_frame])

                # if len(result) > 1:
                #     result_list.append(result[0])
                #     save_frame.append(frame)
                #     thres_lst.append(result[1])

                if len(result) > 0:
                    print('=====================')
                    print(result)
                    print('=====================')
                    result_list.append([result, frame])
                    save_frame.append(frame)

            if id == 5:
                # save_frame_idx = det_ocr_car.selectSave(thres_lst)

                # 存置信度最高的一张图final_save
                # ultra_result = det_ocr.getstr(result_list)
                # ultra_result = result_list[save_frame_idx]
                # else:
                # ultra_result = det_ocr.getstr(result_list)
                # if(len(result_list)>3):
                #     ultra_result, save_frame_idx = det_ocr_car.get_finalResult(result_list[:3])
                # else:
                ultra_result, save_frame_idx = det_ocr_car.get_finalResult(result_list)
                if len(save_frame) > 0:
                    final_save = save_frame[save_frame_idx]
                    cv2.imwrite('tmp/chepai.jpg', final_save)
                with open("0611-001-reccar.txt", "a") as fw:
                    fw.write(str(cont_num) + "/t" + ultra_result + "/n")
                    cont_num += 1
                self.recMessage.emit(1, ultra_result)
                result_list = []
                save_frame = []
                thres_lst = []
                print("ultra rec result:", ultra_result)
                # 写入结果到文件中
                file_path = 'tmp/rec.txt'
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(ultra_result)
                print("结果已保存到文件:", file_path)


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

        self.ocr_queue = Queue(maxsize=2000)
        self.rec_queue = Queue(maxsize=2000)

        self.ocr_thread = OcrThread(self.ocr_queue)
        self.ocr_thread.ocrMessage.connect(self.update_ocr)

        self.rec_thread = RecThread(self.rec_queue)
        self.rec_thread.recMessage.connect(self.update_rec)

        self.path = "result"
        self.channel = "01"
        self.save_thread = SaveProcessWorker(self.path, self.channel)

        self.image_queues = [Queue(maxsize=2000) for _ in range(len(self.camera_config))]
        self.image_process_workers = [ImageProcessWorker(camera_info, queue, self.ocr_queue) for camera_info, queue in
                                      zip(list(self.camera_config.values())[:3], self.image_queues[:3])]

        for worker in self.image_process_workers:
            worker.image_processed.connect(self.handle_image_processed)
            worker.dataQueued.connect(self.update_label_1)
            worker.frameCaptured.connect(self.update_label)

        self.additional_image_workers = [ImageProcessWorker2(camera_info, queue, self.rec_queue, self.ocr_queue) for camera_info, queue in
                                         zip(list(self.camera_config.values())[3:], self.image_queues[3:])]
        for worker in self.additional_image_workers:
            worker.image_processed.connect(self.handle_image_processed)
            worker.dataQueued.connect(self.update_label_2)

        self.capture_processes = [ImageCaptureProcess(camera_info, queue) for camera_info, queue in zip(self.camera_config.values(), self.image_queues)]
        # 单独设置进程名称
        for i, process in enumerate(self.capture_processes):
            process.name = f"CaptureProcess_{i}"

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
