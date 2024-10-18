# -*- coding: utf-8 -*-
import cv2
import multiprocessing as mp

def save_rtsp_to_mp4(rtsp_url, output_file):
    output_file = f"E:/25tup/{output_file}"
    # 打开 RTSP 流
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("无法打开RTSP流")
        return
    # 获取视频帧的宽度、高度和帧率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(width,height)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(fps)
    # 定义视频编码器和输出文件
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v 编码器
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    if not out.isOpened():
        print("无法打开输出文件")
        return
    print(f"开始录制视频，输出文件: {output_file}")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取帧，可能RTSP流已断开")
            break
        # 将帧写入输出文件
        out.write(frame)
        # 显示帧（可选）
        # cv2.imshow('RTSP Stream', frame)
        # 按 'q' 键退出1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("录制完成，文件已保存")


# # 示例使用
# rtsp_url = "rtsp://admin:Dtd12345++@172.20.71.114:554/Streaming/Channels/101?buffer_size=8192"  # 替换为你的 RTSP 流 URL
# output_file = "output_video.mp4"  # 输出的 MP4 文件路径
# save_rtsp_to_mp4(rtsp_url, output_file)





data = {
    "params_left": {
      "file_path": "rtsp://admin:Dtd12345++@172.20.71.111:554/Streaming/Channels/101?buffer_size=8192",
      "roi": [400, 1300, 50, 950],
      "id": 0,
      "queue": "",
      "direction": "left"
    },
    "params_right": {
      "file_path": "rtsp://admin:Dtd12345++@172.20.71.116:554/Streaming/Channels/101?buffer_size=8192",
      "roi": [400, 1300, 1650, 2550],
      "id": 2,
      "queue": "",
      "direction": "right"
    },
    "params_top": {
      "file_path": "rtsp://admin:Dtd12345++@172.20.71.113:554/Streaming/Channels/101?buffer_size=8192",
      "roi":[250, 1350, 780,1880],
      "id": 1,
      "queue": "",
      "direction": "top"
    },
    "params_front": {
      "file_path": "rtsp://admin:Dtd12345++@172.20.71.112:554/Streaming/Channels/101?buffer_size=8192",
      "roi": [450, 1440, 780, 1700],
      "id": 0,
      "direction": "front"
    },
    "params_rear": {
      "file_path": "rtsp://admin:Dtd12345++@172.20.71.114:554/Streaming/Channels/101?buffer_size=8192",
      "roi": [60, 1440, 530, 1770],
      "id": 1,
      "direction": "rear"
    }
  }






if __name__ == '__main__':
    for key, value in data.items():
        process = mp.Process(target=save_rtsp_to_mp4, args=(value["file_path"], "output_video_"+value["direction"]+".mp4"))
        process.start()