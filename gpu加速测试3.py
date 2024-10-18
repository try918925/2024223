import os
os.add_dll_directory(r"C:/opencv-4.9.0/build/install/x64/vc16/bin")
os.add_dll_directory(r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin")
import cv2
from multiprocessing import Process, Queue
import json
import threading
from datetime import datetime
import time

def time_str(fmt=None):
    if fmt is None:
        fmt = '%Y_%m_%d_%H_%M_%S_%f'
    return datetime.today().strftime(fmt)


class ImageCaptureProcess(Process):
    def __init__(self, camera_info, queue):
        super().__init__()
        # self.file_path = "C:/Users/Install/Desktop/2024223/my_test_img/20240911-175546.mp4"
        self.file_path = r"C:\TianjinGangTest\10181402\left.mp4"
        self.roi = [450, 1440, 780, 1700]
        self.stop_flag = False
        self.queue = queue  # 使用队列来传递数据

    def run(self):
        while not self.stop_flag:
            try:
                decoder = cv2.cudacodec.createVideoReader(self.file_path)
                frame_counter = 0
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
                        # path = f"C:/Users/Install/Desktop/2024223/my_test_img/front/{time_str()}.jpg"
                        # self.queue.put((path, frame))
                        print(frame)
                        print("完成一个:",frame_counter)
                        del frame
            except Exception as e:
                print("Error:", e)
            finally:
                # 不需要显式调用 release
                pass


def writer_img(queue):
    while True:
        path, frame = queue.get()
        try:
            cv2.imwrite(path, frame)
        except Exception as e:
            print(f"Error saving image {path}: {e}")


if __name__ == '__main__':
    with open('camera.json', 'r') as f:
        camera_config = json.load(f)

    queue = Queue()
    imagecaptureprocess = ImageCaptureProcess(camera_config, queue)
    imagecaptureprocess.start()

    # # 启动一个线程来处理图像保存
    # writer_thread = threading.Thread(target=writer_img, args=(queue,))
    # writer_thread.start()

    # # 运行 10 秒后停止
    # time.sleep(10)
    # imagecaptureprocess.stop_flag = True
    # imagecaptureprocess.join()
    #
    # # 发送结束信号到 writer_thread
    # queue.put((None, None))
    # writer_thread.join()
