import cv2
import copy
import numpy as np
from collections import Counter



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
        ########################################################################################################
        self.timelist = []  # 用于存储每张图片的推理时间，计算平均推理时间用

    def process_frame(self, frame):
        img = copy.deepcopy(frame)
        img = np.ascontiguousarray(img)  # as contiguous array将一个内存不连续存储的数组转换为内存连续的数组，使的运行速度更快
        obj_list = []
        obj_list = self.detector.det([img])[0]
        

        if len(obj_list) == 0:  # 场景内无目标
            self.non_truck += 1
        else:  # 场景内有目标
            self.non_truck = 0
            self.new_truck = True  # 场景内有新目标进入
            # print(obj_list)
            for _, class_name, score, p1p2, oxywh in obj_list:
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

                if len(self.max_area_dict) == 0:
                    self.max_area_dict['label'] = class_name
                    self.max_area_dict['area'] = now_area
                    self.max_area_dict['img'] = frame  # 如果要带框的就写img
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

    def get_result(self):
        try:
            return self.max_area_dict
        except Exception:
            return None, None