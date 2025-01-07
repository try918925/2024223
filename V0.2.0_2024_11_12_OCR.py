# -*- coding: utf-8 -*-
import os

os.add_dll_directory(r"C:/opencv-4.9.0/build/install/x64/vc16/bin")
os.add_dll_directory(r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin")
import cv2
import threading
import infer_det_rec as det_ocr
from concurrent.futures import ThreadPoolExecutor


class OcrThread(threading.Thread):
    def __init__(self, folder_path):
        super().__init__()
        self.config_dict = {
            "ocr_det_config": "./config/det/my_det_r50_db++_td_tr.yml",
            "ocr_rec_config": "./config/rec/my_en_PP-OCRv3_rec.yml"
        }
        self.ocr = det_ocr.OCR_process(self.config_dict)
        self.file_path = folder_path
        self.direction = "rear"
        self.no_car = 0
        self.roi = [520, 1425, 480, 1790]
        self.roi_cols = (self.roi[2], self.roi[3])
        self.roi_rows = (self.roi[0], self.roi[1])

    def run(self):
        result_list = []
        result = None
        decoder = cv2.cudacodec.createVideoReader(os.path.join(self.file_path, self.direction + '.mp4'))
        while True:
            ret, frame = decoder.nextFrame()
            if not ret or frame is None:
                break

            frame = frame.colRange(self.roi_cols[0], self.roi_cols[1]).rowRange(self.roi_rows[0], self.roi_rows[1])
            frame = frame.download()  # 裁剪后的图像下载到 CPU
            frame = frame[:, :, :3]  # 裁剪通道

            if self.direction == "left":
                result = self.ocr.process_imgs([frame], "left")
            if self.direction == "right":
                result = self.ocr.process_imgs([frame], "right")
            if self.direction == "front":
                frame_ues = cv2.GaussianBlur(frame, (5, 5), 0)
                result = self.ocr.process_imgs([frame_ues, frame], "front")
            if self.direction == "rear":
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
                self.no_car = 0
            else:
                self.no_car += 1
                if self.no_car > 6:
                    ultra_result_one = det_ocr.vote2res(result_list)
                    result_list = []
                self.write_ocr_data(ultra_result_one)

    def write_ocr_data(self, ultra_result):
        file_path = os.path.join(self.file_path, "ocr.txt")
        # 写入结果到文件中
        with open(file_path, 'a', encoding='utf-8') as file:
            ultra_result = ultra_result[:-4] + ' ' + ultra_result[-4:]
            print("箱号:",ultra_result)
            file.write(ultra_result + '\n')


def process_folder(folder_path):
    print(f"处理中: {folder_path}")
    rec_thread = OcrThread(folder_path)
    rec_thread.start()
    rec_thread.join()  # 等待线程完成


if __name__ == "__main__":
    folder_path = r'D:\TianJinGangTest\22fps_crop'
    subfolder_names = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]

    # 使用线程池来并行处理多个文件夹，设置最大并发数为10
    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(process_folder, [os.path.join(folder_path, name) for name in subfolder_names])
