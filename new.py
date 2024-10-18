import sys
import cv2
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QThread
from multiprocessing import Process, Queue


class ImageCaptureProcess(Process):
    def __init__(self, camera_id, queue):
        super().__init__()
        self.camera_id = camera_id
        self.queue = queue

    def run(self):
        capture = cv2.VideoCapture(self.camera_id)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera_id}")

        while True:
            ret, frame = capture.read()
            if ret:
                self.queue.put((self.camera_id, frame))


class ImageProcessWorker(QThread):
    image_processed = Signal(int, object)

    def __init__(self, queue):
        super().__init__()
        self.queue = queue

    def run(self):
        while True:
            if not self.queue.empty():
                camera_id, image = self.queue.get()
                # 在这里添加您的图像处理代码
                # 例如，简单地发射信号以通知图像已处理
                self.image_processed.emit(camera_id, image)


class MainWindow(QObject):
    def __init__(self):
        super().__init__()
        self.image_queues = [Queue() for _ in range(5)]
        self.capture_processes = [ImageCaptureProcess(camera_id, self.image_queues[i]) for i, camera_id in enumerate(range(5))]
        self.image_process_worker = ImageProcessWorker(self.image_queues[0])
        self.image_process_worker.image_processed.connect(self.handle_image_processed)

    def start(self):
        for process in self.capture_processes:
            process.start()
        self.image_process_worker.start()

    def handle_image_processed(self, camera_id, image):
        # 处理已处理的图像
        print(f"Image processed from camera {camera_id}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.start()
    sys.exit(app.exec())
