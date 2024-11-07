import cv2
import subprocess
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=10)

video_data = "C:/TianJinGangTest/10191630/right.mp4"


def write_frame(frame):
    path = rf"C:\TianjinGangTest\0\{time.time()}.jpg"
    cv2.imwrite(path, frame)


# 使用 ffprobe 获取视频信息
ffprobe_cmd = [
    'ffprobe', '-v', 'error', '-select_streams', 'v:0',
    '-show_entries', 'stream=width,height,r_frame_rate', '-of', 'csv=p=0', video_data
]

output = subprocess.check_output(ffprobe_cmd).decode().strip().split(',')
width, height, r_frame_rate = output
width, height = map(int, (width, height))

# 计算帧率
num, denom = map(int, r_frame_rate.split('/'))
fps = num / denom
target_fps = 25

ffmpeg_cmd = [
    'ffmpeg', '-hwaccel', 'cuda', '-i', video_data,
    '-vf', f'fps={target_fps}', '-f', 'image2pipe', '-pix_fmt', 'bgr24', '-vcodec', 'rawvideo', '-'
]


pipe = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, bufsize=10 ** 8)
print(f"视频分辨率: {width}x{height}, 帧率: {fps:.2f} FPS")

start_time = time.time()
count = 0

while True:
    start_time = time.time()
    raw_image = pipe.stdout.read(width * height * 3)
    if len(raw_image) == 0:
        print("检测到断流，正在重连...")
        pipe.kill()
        time.sleep(1)
        pipe = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, bufsize=10 ** 8)
        continue

    frame = np.frombuffer(raw_image, dtype='uint8').reshape((height, width, 3))

    # executor.submit(write_frame, frame)
    count += 1
    end_time = time.time() - start_time
    # if end_time >= 1:
    print(f"数量 {count}   时间:{end_time}")
        # start_time = time.time()
        # count = 0

pipe.stdout.flush()
cv2.destroyAllWindows()
