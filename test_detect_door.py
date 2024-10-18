import cv2 
import numpy as np
from configs import config_5s_trt_door as my_config
# from configs import config_5s_truck as my_config
from algorithms.detector import YOLOv5Detector
import Container_det as cont_trt_infer

import threading



img = cv2.imread("./1_298.jpg")
img = np.ascontiguousarray(img)  # as contiguous array将一个内存不连续存储的数组转换为内存连续的数组，使的运行速度更快

import time

def process(i):
    detector = YOLOv5Detector.from_config(my_config)
    conDet = cont_trt_infer.container_detect(detector)
    time.sleep(0.001)
    for _ in range(5):
        tik = time.time()
        obj_list = []
        # obj_list = detector.det([img])[0]   
        # print(len(obj_list), obj_list, time.time() - tik)  
        # [500, 1400, 500, 1700]
        conDet.process_frame(img)
        print(i, "===", conDet.non_truck, conDet.new_truck)
        time.sleep(0.001)

for i in range(5):
    t = threading.Thread(target=process, args=(i, ))
    t.start()
    #t.join()



# def draw_bbox(img, p1, p2, color=(0, 255, 0), thickness=None, bbox_message=None, font_scale=0.6):
#     cv2.rectangle(img, p1, p2, color, thickness=thickness)
#     font_thickness = int(thickness*0.75)
#     if bbox_message is not None:
#         t_size = cv2.getTextSize(bbox_message, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness=font_thickness)[0]
#         cv2.rectangle(img, p1, (p1[0] + t_size[0], p1[1] - t_size[1] - 3), color, -1)
#         cv2.putText(img, bbox_message, (p1[0], p1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
#                     (0, 0, 0), font_thickness, lineType=cv2.LINE_AA)
#     return img

# for class_id, class_name, score, p1p2, oxywh in obj_list:
#     # print(class_name, score, p1p2, oxywh)
#     x1, y1, x2, y2 = map(int, p1p2)
#     xo, yo, w, h = oxywh
#     now_area = w * h
#     show_img = draw_bbox(img, (x1, y1), (x2, y2), color=detector._colors[class_id], thickness=3, bbox_message= class_name)

# cv2.imwrite("test1.png", show_img)