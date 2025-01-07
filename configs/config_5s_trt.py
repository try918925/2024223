import os,sys

# ----------------------------------------
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
execution_path = os.path.dirname(CURRENT_DIR)
execution_path = execution_path.replace("\\", "/")
print("execution_path:", execution_path)
oring_path = sys.argv[0]
print("oring_path:", oring_path)
real_path = os.path.realpath(oring_path)
filed_path = os.path.dirname(real_path)

LOG_DIR = "./log"
RESULT_DIR = "./result"
# --------------------------------------------------
YOLO_DEVICE = "cuda:0"

YOLO_CLASSES = ['20', '40']  # notice to be int ['door', 'nodoor']

YOLO_NET_CONF = os.path.join(CURRENT_DIR, 'yolov5s.yaml')
YOLO_WEIGHT_PATH = os.path.join(filed_path, "sxDet.engine")
YOLO_TARGET_SIZE = (640, 640)
YOLO_PADDING_COLOR = (114, 114, 114)
YOLO_THRESHOLD_CONF = 0.75  # 置信度的过滤值
YOLO_THRESHOLD_IOU = 0.5
print("YOLO_WEIGHT_PATH:", YOLO_WEIGHT_PATH)