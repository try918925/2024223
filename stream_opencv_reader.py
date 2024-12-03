import cv2
import numpy as np


class stream_opencv_reader(object):
    def __init__(self, source, source_type='file', fps=5, resolution=(2560, 1440)):
        '''
        Arguments:
            source: (str) ffmppeg surport 
            source_type: (str) `'file'` or `'stream'`
            fps: notice to < source_fps
        '''
        self.capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG, [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY])
        self.source_type = source_type
        if source_type == 'stream':  # todo: reload read()
            self.capture.set(cv2.CAP_PROP_FPS, fps)
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        elif source_type == 'file':
            source_fps = int(self.capture.get(cv2.CAP_PROP_FPS))
            print(f"source_fps: {source_fps}")
            self.frame_intervals = np.zeros(source_fps, dtype=int)
            interval = source_fps / fps
            for i in range(fps):
                index = int(i * interval) % source_fps
                self.frame_intervals[index] = 1
            self.cnt = 0

    def read(self, ):
        if self.source_type == 'file':
            while self.frame_intervals[self.cnt] == 0:
                self.capture.grab()
                self.cnt += 1
                if not self.cnt < len(self.frame_intervals):
                    self.cnt = 0

            self.capture.grab()
            ret, frame = self.capture.retrieve()
            self.cnt += 1
            if not self.cnt < len(self.frame_intervals):
                self.cnt = 0

            return ret, frame

        elif self.source_type == 'stream':
            ret = self.capture.grab()
            ret, frame = self.capture.retrieve()
            return ret, frame

    def seek(self, ):
        '''
            todo
        '''
        pass


if __name__ == '__main__':
    import time

    # file_path = 'E:\\workspace\\tianji\\vedio_seg\\101\\ch0001_20240912T092756Z_20240912T095147Z_X02010007278000000.mp4'
    file_path = r'C:\TianjinGangTest\10.28 01\right.mp4'
    vedio_reader = stream_opencv_reader(source=file_path, fps=15)
    start_time = time.time()
    count = 0
    while True:
        time.sleep(0.02)
        ret, frame = vedio_reader.read()
        if not ret:
            break
        count += 1
        end_time = time.time() - start_time
        if end_time >= 1:
            print(f"数量：{count},时间：{end_time}")
            start_time = time.time()
            count = 0
