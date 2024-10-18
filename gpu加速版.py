import os
import time

os.add_dll_directory(r"C:/opencv_build_data/opencv-4.9.0/build/install/x64/vc16/bin")
os.add_dll_directory(r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin")
import cv2
from multiprocessing import Process, Queue, Lock
import json
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor


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
        self.dsize = (1066, 600)
        self.roi_cols = (self.roi[2], self.roi[3])
        self.roi_rows = (self.roi[0], self.roi[1])

    def run(self):
        try:
            decoder = cv2.cudacodec.createVideoReader(self.file_path)
            frame_count = 0
            start_time = time.time()

            while True:
                ret, frame = decoder.nextFrame()
                if not ret or frame is None:
                    decoder = cv2.cudacodec.createVideoReader(self.file_path)
                    print(f"{self.direction}: 图片断流")
                    continue

                # 传递帧
                self.transfer_frame_gpu(frame)

                # 增加帧计数
                frame_count += 1

                # 计算当前时间和开始时间的差
                elapsed_time = time.time() - start_time

                # 每秒统计一次帧数
                if elapsed_time >= 1.0:
                    print(f"{self.direction}: 每秒传输 {frame_count} 帧")
                    # 重置计数器和开始时间
                    frame_count = 0
                    start_time = time.time()
        except Exception as e:
            print(f"ImageCaptureProcess---{self.direction}: 图片读取有问题, 错误: {e}")

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
            # logger.error(f"transfer_frame_gpu错误:{error}")
            print(f"transfer_frame_gpu错误:{error}")

    def transfer_frame_one(self, frame):
        try:
            frame_cpu = frame.download()  # 裁剪后的图像下载到 CPU
            # print(frame_cpu.shape)
            frame_cpu = frame_cpu[:, :, :3]  # 裁剪通道
            self.queue.put((self.camera_id, frame_cpu))
        except Exception as error:
            # logger.error(self.direction, "错误:", error)
            print(self.direction, "错误:", error)


class ImageProcessWorker(Process):
    def __init__(self, camera_info, qu1, qu2):
        super().__init__()
        self.camera_id = camera_info['id']
        self.file_path = camera_info['file_path']
        self.direction = camera_info['direction']
        self.roi = camera_info.get('roi', None)
        self.queue = qu1
        self.ocr_queue = qu2

    def run(self):
        print(self.direction, "ImageProcessWorker")
        while True:
            start_time = time.time()
            if not self.queue.empty():
                camera_id, frame, seg_frame = self.queue.get()
                # camera_id, frame = self.queue.get()
                # del camera_id
                # del frame
                # frame = cv2.resize(frame, ((2560, 1440)), interpolation=cv2.INTER_CUBIC)
                # path = f"C:/Users/Install/Desktop/2024223/my_test_img/{self.direction}/{time_str()}.jpg"
                # grabber = threading.Thread(target=self.writer_img, args=(path, frame))
                # grabber.start()

                # print(self.direction,":",frame.shape)
                # seg_frame = frame[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
                print(self.direction, f":ImageProcessWorker 队列大小为：{self.queue.qsize()}")
                print(f"{self.direction}相机:完成算法结果时间:", time.time() - start_time, "\n")

    def writer_img(self, path, frames):
        cv2.imwrite(path, frames)


class ImageProcessWorker2(Process):
    def __init__(self, camera_info, qu1, qu2, qu3):
        super().__init__()
        self.camera_id = camera_info['id']
        self.file_path = camera_info['file_path']
        self.direction = camera_info['direction']
        self.roi = camera_info.get('roi', None)
        self.queue = qu1
        self.rec_queue = qu2
        self.ocr_queue = qu3

    def run(self):
        print(self.direction, "ImageProcessWorker2")
        while True:
            try:
                if not self.queue.empty():
                    start_time = time.time()
                    camera_id, frame = self.queue.get()
                    # del camera_id
                    # del frame
                    print(self.direction, f":ImageProcessWorker2 队列大小为：{self.queue.qsize()}")
                    print(f"{self.direction}相机:完成算法结果时间:", time.time() - start_time, "\n")
            except Exception as error:
                print("ImageProcessWorker2:", error)


if __name__ == '__main__':
    with open('camera_test.json', 'r') as f:
        camera_config = json.load(f)

    image_queues = [Queue() for _ in range(len(camera_config))]

    ocr_queue = Queue()
    rec_queue = Queue()

    image_process_workers = [ImageProcessWorker(camera_info, queue, ocr_queue) for camera_info, queue in
                             zip(list(camera_config.values())[:3], image_queues[:3])]

    additional_image_workers = [ImageProcessWorker2(camera_info, queue, rec_queue, ocr_queue) for camera_info, queue in
                                zip(list(camera_config.values())[3:], image_queues[3:])]

    # lock = [Lock() for _ in range(len(camera_config))]

    captureprocesses = [ImageCaptureProcess(camera_info, queue) for camera_info, queue in zip(camera_config.values(), image_queues)]

    for image_process in image_process_workers:
        image_process.start()

    for image_process in additional_image_workers:
        image_process.start()

    for captureproc in captureprocesses:
        captureproc.start()
