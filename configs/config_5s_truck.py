import os
import cv2
# ----------------------------------------
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
LOG_DIR = "./log"
RESULT_DIR = "./result"

## 相机设置
VEDIO_CAM = True # 使用本地视频代替相机
CAM_LIST = ["cam_20", "cam_22"]
video_path = "20230628-004703.315-rear.mp4"
video = cv2.VideoCapture(video_path)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频的宽度（单位：像素）
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频的高度（单位：像素）

if VEDIO_CAM:
    CAMERA_DICT = {
        "cam_22":{
            "ip":video_path,
            "comment": "最大箱面检测",
            "resolution": (width, height),  # 相机分辨率
            "show_process":False # 是否需要展示过程
        },
        "cam_20":{
            "ip":video_path,
            "comment": "最大箱面检测",
            "resolution": (width, height),  # 相机分辨率
            "show_process":False # 是否需要展示过程
        },
    }
else:
    CAMERA_DICT = {
        "cam_22": {
                "ip": "192.168.9.22", 
                "comment": "全景判断位置",
                "username": "admin", 
                "password": "jskj0312",
                "resolution": (3840, 2160),  # 相机分辨率
                "gpu_id": 0,
                "allow_disconnected": False,
                "show_process":True # 是否需要展示过程
            },
        "cam_20": {
                "ip": "192.168.9.20", 
                "comment": "小车下判断位置",
                "username": "admin", 
                "password": "jskj0312",
                "resolution": (2560, 1440),  # 相机分辨率
                "gpu_id": 0,
                "allow_disconnected": False,
                "show_process":True # 是否需要展示过程
            },
}
    


# --------------------------------------------------
YOLO_DEVICE = "cuda:0"
DATA_YAML = "./configs/truck_data.yaml"
YOLO_CLASSES = ['door', 'nodoor'] # # notice to be int

YOLO_NET_CONF = os.path.join(CURRENT_DIR, 'yolov5s.yaml')
YOLO_WEIGHT_PATH = 'best.pt'
YOLO_TARGET_SIZE = (640, 640)
YOLO_PADDING_COLOR = (114, 114, 114)
YOLO_THRESHOLD_CONF = 0.7                # 置信度的过滤值
YOLO_THRESHOLD_IOU = 0.5

