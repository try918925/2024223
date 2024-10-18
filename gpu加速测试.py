import os
os.add_dll_directory(r"C:/opencv-4.9.0/build/install/x64/vc16/bin")
os.add_dll_directory(r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin")
import cv2
from multiprocessing import Process, Queue
import json
import torch
torch.cuda.set_per_process_memory_fraction(0.1, 0)
import time

class ImageCaptureProcess(Process):
    def __init__(self, camera_info, queue):
        super().__init__()
        # self.file_path = "rtsp://admin:Dtd12345++@172.20.71.111:554/Streaming/Channels/101"
        self.file_path = r"C:\TianjinGangTest\10181402\front.mp4"
        self.roi = [450, 1440, 780, 1700]
        self.stop_flag = False
        self.queue = queue  # 使用队列来传递数据

    def run(self):
        while not self.stop_flag:
            try:
                decoder = cv2.cudacodec.createVideoReader(self.file_path,)
                frame_counter = 0
                count = 0
                start_time = time.time()
                while not self.stop_flag:
                    ret, frame = decoder.nextFrame()
                    if not ret or frame is None:
                        # 视频播放完毕，重新开始
                        # decoder = cv2.cudacodec.createVideoReader(self.file_path)
                        print("相机 读取图片失败")
                        continue
                    frame = frame.download()
                    frame = frame[:, :, :3]
                    frame_counter += 1
                    if frame_counter % 2 == 1:
                        print(frame.shape)
                        print("完成一个:",frame_counter)
                        # del frame
                        # count += 1
                        # if time.time() - start_time >= 1:
                        #     print("count的值为:", count)
            except Exception as e:
                print("Error:", e)
            finally:
                pass

if __name__ == '__main__':
    with open('camera.json', 'r') as f:
        camera_config = json.load(f)

    queue = Queue()
    imagecaptureprocess = ImageCaptureProcess(camera_config, queue)
    imagecaptureprocess.start()