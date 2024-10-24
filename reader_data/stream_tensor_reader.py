import torch
import torchaudio
import torchvision

from torchaudio.io import StreamReader
from torchvision.transforms import Resize

class stream_tensor_reader(object):
    def __init__(self, soruce, frames_per_chunk=1, decoder='hevc_cuvid', hw_accel='cuda:0' ):
        '''
        Arguments:
            soruce, (str): (file-like object or Tensor), torchaudio.io.StreamReader src surpport
            frames_per_chunk, (int): 每次读取图片数量, default `1`
            decoder, (str): 解码器, default `ffmpeg_utils.get_video_decoders().keys()`
            hw_accel, (str): 硬件加速, default `cuda:0`
        '''
        self.capture = StreamReader(src= soruce)        
        self.capture.add_video_stream(
            frames_per_chunk= frames_per_chunk,  
            decoder= decoder, 
            hw_accel= hw_accel
        )        
        # self.trans_resize = Resize(img_size)

    def read_tensor_img(self, ):
        '''
        Returns:
            img_tensor: HCWH, (rgb)
        '''
        EOF = self.capture.fill_buffer()
        if EOF == 0:
            frames = self.capture.pop_chunks()  # NCHW yuv
            frame = frames[0].to(torch.float)

            # frame = self.trans_resize(frame)
           
            y = frame[..., 0, :, :]
            u = frame[..., 1, :, :]
            v = frame[..., 2, :, :]  
            r = y + 1.14 * v
            g = y + -0.396 * u - 0.581 * v
            b = y + 2.029 * u
            
            rgb = torch.stack([r, g, b], -1)  
            img_tensor = rgb.permute((0, 3, 1, 2))
            return True, img_tensor
        elif EOF == 1: # end of stream
            return False, None

    def seek(self, ):
        '''
         todo
        '''
        pass    

    def read_np_img(self, ):
        '''
        Returns:
            img_np: CHW, bgr
        '''
        EOF = self.capture.fill_buffer()
        if EOF == 0:
            frames = self.capture.pop_chunks()  # NCHW yuv
            frame = frames[0].to(torch.float)
            # frame = self.trans_resize(frame)
            print(frame.shape, frame.dtype, frame.device)

            y = frame[..., 0, :, :]
            u = frame[..., 1, :, :]
            v = frame[..., 2, :, :]

            y /= 255
            u = u / 255 - 0.5
            v = v / 255 - 0.5

            r = y + 1.14 * v
            g = y + -0.396 * u - 0.581 * v
            b = y + 2.029 * u

            bgr = torch.stack([b, g, r], -1)  
            bgr = (bgr * 255).clamp(0, 255).to(torch.uint8)
            print(bgr.shape, bgr.dtype, bgr.device)
            img_np = bgr.cpu().numpy()
            return True, img_np
        elif EOF == 1: # end of stream
            return False, None

if __name__ == '__main__':
    import time
    import cv2
    resize_640 = Resize((640, 640))
    
    file_path = 'E:\\workspace\\tianji\\vedio_seg\\101\\ch0001_20240912T092756Z_20240912T095147Z_X02010007278000000.mp4'

    # (username, password, hostname) = ('admin', 'Dnt@QC2023', '10.141.1.101')
    # file_path = f"rtsp://{username}:{password}@{hostname}:554/Streaming/Channels/101"

    vedio_reader = stream_tensor_reader(file_path)

    for i in range(1000):
        ret, img = vedio_reader.read_tensor_img()
        img = resize_640(img)
        print(time.time(), img.shape, img.dtype,)
        time.sleep(0.03)

    # for i in range(100):
    #     ret, img = vedio_reader.read_np_img()
    #     # img = resize_640(img)
    #     # print(img.shape, img.dtype,)
    #     cv2.imwrite('./test.jpg', img[0])
    #     time.sleep(0.03)
        