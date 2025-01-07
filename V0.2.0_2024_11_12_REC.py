import os
os.add_dll_directory(r"C:/opencv-4.9.0/build/install/x64/vc16/bin")
os.add_dll_directory(r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin")
import cv2
from concurrent.futures import ProcessPoolExecutor
import infer_det_rec_car as det_ocr_car


class RecProcess:
    def __init__(self, file_path):
        self.weights_dict = {
            "ocr_det_config": "./config_car/det/my_car_det_r50_db++_td_tr.yml",
            "ocr_rec_config": "./config_car/rec/my_rec_chinese_lite_train_v2.0.yml"
        }
        self.lp = det_ocr_car.OCR_process(self.weights_dict)
        self.file_path = file_path
        self.no_car = 0
        self.roi = [450, 1440, 780, 1700]
        self.roi_cols = (self.roi[2], self.roi[3])
        self.roi_rows = (self.roi[0], self.roi[1])

    def run(self):
        result_list = []
        save_frame = []
        decoder = cv2.cudacodec.createVideoReader(os.path.join(self.file_path, 'front.mp4'))

        while True:
            ret, frame = decoder.nextFrame()
            if not ret or frame is None:
                break

            frame = frame.colRange(self.roi_cols[0], self.roi_cols[1]).rowRange(self.roi_rows[0], self.roi_rows[1])
            frame = frame.download()  # 裁剪后的图像下载到 CPU
            frame = frame[:, :, :3]  # 裁剪通道

            use_frame = cv2.GaussianBlur(frame, (5, 5), 0)
            result = self.lp.process_imgs([use_frame])
            if len(result) > 0:
                self.no_car = 0
                result_list.append([result, frame])
                save_frame.append(frame)
            else:
                self.no_car += 1
                if self.no_car >= 5:
                    self.process_results(result_list, save_frame)

    def process_results(self, result_list, save_frame):
        ultra_result, save_frame_idx = det_ocr_car.get_finalResult(result_list)
        if len(save_frame) > 0:
            final_save = save_frame[save_frame_idx]
            self.writer_data(ultra_result, final_save)
            result_list.clear()
            save_frame.clear()

    def writer_data(self, ultra_result, frame):
        """写入识别结果和图像"""
        file_path = os.path.join(self.file_path, "rec.txt")
        image_path = os.path.join(self.file_path, f'{ultra_result}_chepai.jpg')

        try:
            with open(file_path, 'a', encoding='utf-8') as file:
                file.write(ultra_result + '\n')
            cv2.imwrite(image_path, frame)
            print(f"写入成功: {ultra_result}，图像已保存")
        except Exception as e:
            print(f"写入文件时出错: {e}")


def process_folder(folder_path):
    print(f"处理中: {folder_path}")
    rec_process = RecProcess(folder_path)
    rec_process.run()


if __name__ == "__main__":
    folder_path = r'D:\TianJinGangTest\22fps_crop'
    subfolder_names = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]

    # 使用进程池来并行处理多个文件夹，设置最大并发数为10
    with ProcessPoolExecutor(max_workers=10) as executor:
        executor.map(process_folder, [os.path.join(folder_path, name) for name in subfolder_names])