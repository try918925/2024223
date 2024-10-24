import os

# ----------------------------------------
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
LOG_DIR = "./log"
RESULT_DIR = "./result"
# --------------------------------------------------
YOLO_DEVICE = "cuda:0"

YOLO_CLASSES = ['door', 'nodoor'] # notice to be int ['door', 'nodoor']

YOLO_NET_CONF = os.path.join(CURRENT_DIR, 'yolov5s.yaml')
YOLO_WEIGHT_PATH = 'C:/Users/Install/Desktop/2024223/doorBest.engine'
YOLO_TARGET_SIZE = (640, 640)
YOLO_PADDING_COLOR = (114, 114, 114)
YOLO_THRESHOLD_CONF = 0.72                # 置信度的过滤值
YOLO_THRESHOLD_IOU = 0.35
