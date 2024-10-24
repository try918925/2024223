
import cv2


class stream_opencv_reader(object):
    def __init__(self, source):
        '''
        Arguments:
            source: (str) ffmppeg surport 
        '''
        self.capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG, \
                           [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY])

    def read(self, ):
        ret, frame = self.capture.read()
        return ret, frame

    def seek(self, ):
        '''
            todo
        '''
        pass

if __name__ == '__main__':
    import time
    file_path = 'E:\\workspace\\tianji\\vedio_seg\\101\\ch0001_20240912T092756Z_20240912T095147Z_X02010007278000000.mp4'
    vedio_reader = stream_opencv_reader(file_path)

    for i in range(1000):
        img = vedio_reader.read()
        print(time.time())
        # time.sleep(0.03)