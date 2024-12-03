from configs import config_5s_trt
from algorithms.detector import YOLOv5Detector
import ctypes
import Container_det_trt_yolov5 as cont_trt_infer

PLUGIN_LIBRARY = "./myplugins.dll"
# engine_file_path = "truck_old.engine"
engine_file_path = "truck_old.engine"
ctypes.CDLL(PLUGIN_LIBRARY)
csd_detector = cont_trt_infer.CSD_Detector(engine_file_path)  # 初始化detector
my_container_detect = cont_trt_infer.container_detect(csd_detector)