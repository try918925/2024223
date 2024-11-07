import time
import av
import cv2
import numpy as np
video_path = "C:/TianJinGangTest/10191630/right.mp4"
# 打开视频文件
container = av.open(video_path)
target_fps = 10  # 设置目标帧率
frame_duration = 1.0 / target_fps  # 计算每帧的持续时间
start_time = time.time()
count = 0
for frame in container.decode(video=0):
    # 将帧转换为ndarray以便使用OpenCV
    img = frame.to_ndarray(format='bgr24')
    count += 1
    current_time = time.time()
    # 如果处理帧的时间小于目标帧时间，则等待
    elapsed_time = current_time - start_time
    if elapsed_time < frame_duration:
        time.sleep(frame_duration - elapsed_time)
    end_time = time.time() - start_time
    if end_time >= 1:
        start_time = current_time
        # 显示处理的帧数和时间
        print(f"数量 {count}   时间:{elapsed_time:.3f}")
    # 等待按键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
