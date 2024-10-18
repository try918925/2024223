"""
An example that uses TensorRT's Python api to make inferences.
"""
import ctypes
import os
import shutil
import random
import sys
import time
import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import copy
from collections import Counter

import warnings
warnings.filterwarnings("ignore")

CONF_THRESH = 0.91  # 检测置信度
IOU_THRESHOLD = 0.4
LEN_ONE_RESULT = 38

# Detector类，主要方法为infer(self, img)，输入数据为图像数据，返回obj_list
# 其中，obj_list是包含了类别名、置信度、框x1y1x2y2、框xywh的元组
class CSD_Detector():

    def __init__(self, engine_file_path):
        self.ctx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
            print(engine)
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []
        self.categories = ["door", "nodoor"]

        for binding in engine:
            # print('bingding:', binding, engine.get_binding_shape(binding))
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(cuda_mem))
            if engine.binding_is_input(binding):
                self.input_w = engine.get_binding_shape(binding)[-1]
                self.input_h = engine.get_binding_shape(binding)[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        self.batch_size = engine.max_batch_size

    def infer(self, img):
        # print('batch_size', self.batch_size)
        # Make self the active context, pushing it on top of the context stack.
        self.ctx.push()
        # Restore
        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings
        # Do image preprocess
        batch_input_image = np.empty(shape=[self.batch_size, 3, self.input_h, self.input_w])

        input_image, image_raw, origin_h, origin_w = self.preprocess_image(img)
        np.copyto(batch_input_image[0], input_image)

        batch_input_image = np.ascontiguousarray(batch_input_image)

        np.copyto(host_inputs[0], batch_input_image.ravel())
        start = time.time()
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        context.execute_async(batch_size=self.batch_size, bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        stream.synchronize()
        end = time.time()
        self.ctx.pop()
        output = host_outputs[0]
        # Do postprocess
        result_boxes, result_scores, result_classid = self.post_process(output, origin_h, origin_w)

        obj_list = []
        for j in range(len(result_boxes)):
            box = result_boxes[j]
            class_name = self.categories[int(result_classid[j])]
            conf_score = result_scores[j]
            box = map(int, box)
            x1, y1, x2, y2 = box
            xo, yo, w, h = round((x1 + x2) / 2), round((y1 + y2) / 2), (x2 - x1), (y2 - y1)
            obj_list.append((class_name, conf_score, (x1, y1, x2, y2), (xo, yo, w, h)))
        return obj_list,end-start

    def destroy(self):
        self.ctx.pop()

    def preprocess_image(self, raw_bgr_image):
        image_raw = raw_bgr_image
        h, w, c = image_raw.shape
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        r_w = self.input_w / w
        r_h = self.input_h / h
        if r_h > r_w:
            tw = self.input_w
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((self.input_h - th) / 2)
            ty2 = self.input_h - th - ty1
        else:
            tw = int(r_h * w)
            th = self.input_h
            tx1 = int((self.input_w - tw) / 2)
            tx2 = self.input_w - tw - tx1
            ty1 = ty2 = 0
        image = cv2.resize(image, (tw, th))
        image = cv2.copyMakeBorder(
            image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, None, (128, 128, 128)
        )
        image = image.astype(np.float32)
        image /= 255.0
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = np.ascontiguousarray(image)
        return image, image_raw, h, w

    def xywh2xyxy(self, origin_h, origin_w, x):
        y = np.zeros_like(x)
        r_w = self.input_w / origin_w
        r_h = self.input_h / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            y /= r_h
        return y

    def post_process(self, output, origin_h, origin_w):
        num = int(output[0])
        pred = np.reshape(output[1:], (-1, LEN_ONE_RESULT))[:num, :]
        pred = pred[:, :6]
        # Do nms
        boxes = self.non_max_suppression(pred, origin_h, origin_w, conf_thres=CONF_THRESH, nms_thres=IOU_THRESHOLD)
        result_boxes = boxes[:, :4] if len(boxes) else np.array([])
        result_scores = boxes[:, 4] if len(boxes) else np.array([])
        result_classid = boxes[:, 5] if len(boxes) else np.array([])
        return result_boxes, result_scores, result_classid

    def bbox_iou(self, box1, box2, x1y1x2y2=True):
        if not x1y1x2y2:
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        else:
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)
        inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None) * \
                     np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0, None)
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

        return iou

    def non_max_suppression(self, prediction, origin_h, origin_w, conf_thres=0.9, nms_thres=0.4):
        boxes = prediction[prediction[:, 4] >= conf_thres]
        boxes[:, :4] = self.xywh2xyxy(origin_h, origin_w, boxes[:, :4])
        boxes[:, 0] = np.clip(boxes[:, 0], 0, origin_w - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, origin_w - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, origin_h - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, origin_h - 1)
        confs = boxes[:, 4]
        boxes = boxes[np.argsort(-confs)]
        keep_boxes = []
        while boxes.shape[0]:
            large_overlap = self.bbox_iou(np.expand_dims(boxes[0, :4], 0), boxes[:, :4]) > nms_thres
            label_match = boxes[0, -1] == boxes[:, -1]
            invalid = large_overlap & label_match
            keep_boxes += [boxes[0]]
            boxes = boxes[~invalid]
        boxes = np.stack(keep_boxes, 0) if len(keep_boxes) else np.array([])
        return boxes

# 调用CSD_Detector进行检测，输入为实例化的detector，返回视频中箱面面积最大时刻的图像帧
class container_detect(object):
    def __init__(self,detector):
        self.detector = detector
        self.debug_show = False

        ########################################################################################################
        # 计算没有检测到关键帧的时间，假设1秒30帧，如果10秒（300帧）没有检测到关键帧则认为正在检测的卡车已经走出监控区域，清空字典准备检测下一辆
        self.non_truck = 0
        self.new_truck = False
        # 图像字典，用于存储可能为面积最大的图像信息，max_area_dict = {'label':label,'area':area,'img':array_of_img}
        self.max_area_dict = {}

        self.res_dict = {}
        ########################################################################################################
        self.timelist = []  # 用于存储每张图片的推理时间，计算平均推理时间用

    def door_label_vote(self, lst):
        labels = [item['label'] for item in lst]
        areas = [item['area'] for item in lst]
        images = [item['img'] for item in lst]
        # scores = [item['score'] for item in lst]

        labels_counts = Counter(labels)
        final_label, max_count = labels_counts.most_common(1)[0]

        filtered_areas = [area for label, area in zip(labels, areas) if label == final_label]
        filtered_images = [img for label, img in zip(labels, images) if label == final_label]

        score_areas = [int(area) if area != '' else 0 for area in filtered_areas]
        
        score_max_index = score_areas.index(max(score_areas))
        sorted_indexes = sorted(range(len(score_areas)), key=lambda i: score_areas[i], reverse=True)
        # print('test')
        # print(score_areas[score_max_index])
        # print('test')
        # if len(sorted_indexes) > 7:
        #     top_five_indexes = sorted_indexes[:7]
        # else:
        #     top_five_indexes = sorted_indexes

        save_img = filtered_images[score_max_index]

        # for i in top_five_indexes:
        #     ocr_use_img.append(filtered_images[i])

        return final_label, save_img


    def process_frame(self, frame):
        img = copy.deepcopy(frame)
        img = np.ascontiguousarray(img)  # as contiguous array将一个内存不连续存储的数组转换为内存连续的数组，使的运行速度更快
        obj_list = []
        obj_list,time = self.detector.infer(img)
        self.timelist.append(time)

        if len(obj_list) == 0:  # 场景内无目标
            self.non_truck += 1
        else:  # 场景内有目标
            # print(obj_list)
            self.non_truck = 0
            self.new_truck = True  # 场景内有新目标进入
            for class_name, score, p1p2, oxywh in obj_list:
                # print(class_name, score, p1p2, oxywh)
                x1, y1, x2, y2 = map(int, p1p2)
                xo, yo, w, h = oxywh
                now_area = w * h
                show_img = self.draw_bbox(img, (x1, y1), (x2, y2), thickness=3, bbox_message=class_name)
                if self.debug_show:  # 是否需要展示判断过程
                    show_img = copy.deepcopy(img)
                    cv2.namedWindow("show", 0)
                    cv2.resizeWindow("show", 960, 540)
                    cv2.imshow("show", show_img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                # ##############################留存最大面积的图片#################################
                self.res_dict['label'] = class_name
                self.res_dict['area'] = now_area
                self.res_dict['img'] = frame 
                self.res_dict['score'] = score
                # print(class_name, score, now_area)

                if len(self.max_area_dict) == 0:
                    self.max_area_dict['label'] = class_name
                    self.max_area_dict['area'] = now_area
                    self.max_area_dict['img'] = frame  # 如果要带框的就写img
                
                # 增加判断箱门识别的label是否对应，避免误判
                elif len(self.max_area_dict) != 0:
                    if self.max_area_dict['area'] < now_area:
                        self.max_area_dict['label'] = class_name
                        self.max_area_dict['area'] = now_area
                        self.max_area_dict['img'] = frame  # 如果要带框的就写img
                # ###############################################################################

        # 判断完一帧之后，反馈状态

    def draw_bbox(sel, img, p1, p2, color=(0, 255, 0), thickness=None, bbox_message=None, font_scale=0.6):
        cv2.rectangle(img, p1, p2, color, thickness=thickness)
        font_thickness = int(thickness * 0.75)
        if bbox_message is not None:
            t_size = cv2.getTextSize(bbox_message, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness=font_thickness)[0]
            cv2.rectangle(img, p1, (p1[0] + t_size[0], p1[1] - t_size[1] - 3), color, -1)
            cv2.putText(img, bbox_message, (p1[0], p1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (0, 0, 0), font_thickness, lineType=cv2.LINE_AA)
        return img

    def get_result(self):
        try:
            return self.max_area_dict, self.res_dict
        except Exception:
            return None


if __name__ == "__main__":
    # 加载插件和预训练模型
    PLUGIN_LIBRARY = "./myplugins.dll"
    engine_file_path = "truck.engine"

    ctypes.CDLL(PLUGIN_LIBRARY)

    categories = ["door", "nodoor"]  # 类别名
    csd_detector = CSD_Detector(engine_file_path)  # 初始化detector
    print("Succeed to init case surface detector.")

    my_container_detect = container_detect(csd_detector)
    

    img = cv2.imread(r'C:\Users\Install\Desktop\2024223\result\01\240430\CAR0249\Transfer\01_240430_112410_CAR0249_Transfer_rear_00.jpg')
    my_container_detect.process_frame(img)
    reuslt_dict = my_container_detect.get_result()

    print(reuslt_dict)
 
    csd_detector.destroy()

    # try:
    #     my_container_detect = container_detect(csd_detector)
    #     # # 使用逻辑：
    #     cam_device = cv2.VideoCapture("./video/rear/rear.mp4")
    #     while True:
    #         ret, frame = cam_device.read()

    #         if ret:
    #             my_container_detect.process_frame(frame)
    #             print(my_container_detect.new_truck, my_container_detect.non_truck,
    #                   my_container_detect.max_area_dict.keys())
    #             if my_container_detect.non_truck > 50 and my_container_detect.new_truck:
    #                 # 当连续20帧(约2s)没有集装箱面，且之前有卡车进入时，获取前一段时间面积最大帧
    #                 reuslt_dict = my_container_detect.get_result()
    #                 print(reuslt_dict["label"])  # 关键帧是箱门面 / 非箱门面 ['door', 'nodoor']
    #                 # 需要结合相机判断集装箱朝向
    #                 cv2.imwrite("./result.jpg", reuslt_dict["img"])  # 获取关键帧
    #                 # !!! 获取最大面积图像后刷新是否有车的状态、刷新存下的结果
    #                 my_container_detect.new_truck = False
    #                 my_container_detect.max_area_dict.clear()
    #         else:
    #             reuslt_dict = my_container_detect.get_result()
    #             cv2.imwrite("./result.jpg", reuslt_dict["img"])
    #             # print('mean time',sum(my_container_detect.timelist) / len(my_container_detect.timelist))  # 测试推理时间
    #             break

    #     # cam_device = cv2.VideoCapture("./test_data/20230628-001924.729-front.mp4")
    #     # while True:
    #     #     ret, frame = cam_device.read()

    #     #     if ret:
    #     #         my_container_detect.process_frame(frame)
    #     #         print(my_container_detect.new_truck, my_container_detect.non_truck,
    #     #               my_container_detect.max_area_dict.keys())
    #     #         if my_container_detect.non_truck > 50 and my_container_detect.new_truck:

    #     #             # 当连续20帧(约2s)没有集装箱面，且之前有卡车进入时，获取前一段时间面积最大帧
    #     #             reuslt_dict = my_container_detect.get_result()
    #     #             print(reuslt_dict["label"])  # 关键帧是箱门面 / 非箱门面 ['door', 'nodoor']
    #     #             # 需要结合相机判断集装箱朝向
    #     #             cv2.imwrite("./result.jpg", reuslt_dict["img"])  # 获取关键帧
    #     #             # !!! 获取最大面积图像后刷新是否有车的状态、刷新存下的结果
    #     #             my_container_detect.max_area_dict.clear()
    #     #             my_container_detect.new_truck = False
    #     #     else:
    #     #         break
    # finally:
    #     csd_detector.destroy()
