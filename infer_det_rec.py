from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import os
import sys
import json
# from  numba import cuda
import psutil
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import paddle
#import subprocess
import tracemalloc
from ppocr.data import create_operators, transform
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model
from ppocr.utils.utility import get_image_file_list
# import tools.program as program
import cv2
import yaml
import ocr as ocr_check
from collections import Counter
from itertools import zip_longest
# try:
#     from OCR_TOOL import ocr as ocr_check
# except:
#     from .OCR_TOOL import ocr as ocr_check
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def draw_det_res(dt_boxes, config, img, img_name, save_path):
    
    src_im = img
    for box in dt_boxes:
        box = np.array(box).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = os.path.join(save_path, os.path.basename(img_name))
    cv2.imwrite(save_path, src_im)

def load_config(file_path):
    """
    Load config from yml/yaml file.
    Args:
        file_path (str): Path of the config file to be loaded.
    Returns: global config
    """
    _, ext = os.path.splitext(file_path)
    assert ext in ['.yml', '.yaml'], "only support yaml files for now"
    config = yaml.load(open(file_path, 'rb'), Loader=yaml.Loader)
    return config

def merge_config(config, opts):
    """
    Merge config into global config.
    Args:
        config (dict): Config to be merged.
    Returns: global config
    """
    for key, value in opts.items():
        if "." not in key:
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value
        else:
            sub_keys = key.split('.')
            assert (
                sub_keys[0] in config
            ), "the sub_keys can only be one of global_config: {}, but get: " \
               "{}, please check your running command".format(
                config.keys(), sub_keys[0])
            cur = config[sub_keys[0]]
            for idx, sub_key in enumerate(sub_keys[1:]):
                if idx == len(sub_keys) - 2:
                    cur[sub_key] = value
                else:
                    cur = cur[sub_key]
    return config

class Detection:
    def __init__(self, config_path):
        # self.config, self.device, self.logger, self.vdl_writer = self.program.preprocess()
        self.config = load_config(config_path)
        self.global_config = self.config['Global']
    
    def load_checkpoint(self):
        self.model = build_model(self.config['Architecture'])

        load_model(self.config, self.model)
        # build post process
        self.post_process_class = build_post_process(self.config['PostProcess'])
        self.transforms = []
        for op in self.config['Eval']['dataset']['transforms']:
            op_name = list(op)[0]
            if 'Label' in op_name:
                continue
            elif op_name == 'KeepKeys':
                op[op_name]['keep_keys'] = ['image', 'shape']
            self.transforms.append(op)
        self.ops = create_operators(self.transforms, self.global_config)
        self.model.eval()
    
    def _predict2box(self, img):
        data = {'image': img}
        batch = transform(data, self.ops)

        images = np.expand_dims(batch[0], axis=0)
        shape_list = np.expand_dims(batch[1], axis=0)
        images = paddle.to_tensor(images)
        preds = self.model(images)
        post_result = self.post_process_class(preds, shape_list)
        boxes = post_result[0]['points']
        return boxes
            
            
    def predict(self, img_list):
        img_box = []
        for img_data in img_list:
            h,w = img_data.shape[:2]
            _, encoded_image = cv2.imencode(".jpg", img_data)
            img = encoded_image.tobytes()
            box = self._predict2box(img)
            img_box.append(box)
            # if(len(box)>0):
            #     new_bbox = []
            #     for bbox in box:
            #         ans = bbox[0][0]
            #         if(ans>w*0.3 and ans<w*0.7):  #划分区域在图片的【0.3 ~ 0.7】中间
            #             new_bbox.append(bbox)
            #     # np.array(new_bbox)
            #     img_box.append(new_bbox)
            # else:
            #     img_box.append(box)
               
        return img_box

class OCR_rec:
    def __init__(self, config_path):
        # self.config, self.device, self.logger, self.vdl_writer = self.program.preprocess()
        self.config = load_config(config_path)
        self.global_config = self.config['Global']
    
    def load_checkpoint(self):
        # build post process
        self.post_process_class = build_post_process(self.config['PostProcess'],
                                                self.global_config)

        # build model
        if hasattr(self.post_process_class, 'character'):
            char_num = len(getattr(self.post_process_class, 'character'))
            if self.config["Architecture"]["algorithm"] in ["Distillation",
                                                    ]:  # distillation model
                for key in self.config["Architecture"]["Models"]:
                    if self.config["Architecture"]["Models"][key]["Head"][
                            "name"] == 'MultiHead':  # multi head
                        out_channels_list = {}
                        if self.config['PostProcess'][
                                'name'] == 'DistillationSARLabelDecode':
                            char_num = char_num - 2
                        if self.config['PostProcess'][
                                'name'] == 'DistillationNRTRLabelDecode':
                            char_num = char_num - 3
                        out_channels_list['CTCLabelDecode'] = char_num
                        out_channels_list['SARLabelDecode'] = char_num + 2
                        out_channels_list['NRTRLabelDecode'] = char_num + 3
                        self.config['Architecture']['Models'][key]['Head'][
                            'out_channels_list'] = out_channels_list
                    else:
                        self.config["Architecture"]["Models"][key]["Head"][
                            "out_channels"] = char_num
            elif self.config['Architecture']['Head'][
                    'name'] == 'MultiHead':  # multi head
                out_channels_list = {}
                char_num = len(getattr(self.post_process_class, 'character'))
                if self.config['PostProcess']['name'] == 'SARLabelDecode':
                    char_num = char_num - 2
                if self.config['PostProcess']['name'] == 'NRTRLabelDecode':
                    char_num = char_num - 3
                out_channels_list['CTCLabelDecode'] = char_num
                out_channels_list['SARLabelDecode'] = char_num + 2
                out_channels_list['NRTRLabelDecode'] = char_num + 3
                self.config['Architecture']['Head'][
                    'out_channels_list'] = out_channels_list
            else:  # base rec model
                self.config["Architecture"]["Head"]["out_channels"] = char_num
        self.model = build_model(self.config['Architecture'])

        load_model(self.config, self.model)
        # print(self.model)

        self.transforms = []
        for op in self.config['Eval']['dataset']['transforms']:
            op_name = list(op)[0]
            if 'Label' in op_name:
                continue
            elif op_name in ['RecResizeImg']:
                op[op_name]['infer_mode'] = True
            elif op_name == 'KeepKeys':
                if self.config['Architecture']['algorithm'] == "SRN":
                    op[op_name]['keep_keys'] = [
                        'image', 'encoder_word_pos', 'gsrm_word_pos',
                        'gsrm_slf_attn_bias1', 'gsrm_slf_attn_bias2'
                    ]
                elif self.config['Architecture']['algorithm'] == "SAR":
                    op[op_name]['keep_keys'] = ['image', 'valid_ratio']
                elif self.config['Architecture']['algorithm'] == "RobustScanner":
                    op[op_name][
                        'keep_keys'] = ['image', 'valid_ratio', 'word_positons']
                else:
                    op[op_name]['keep_keys'] = ['image']
            self.transforms.append(op)
        self.global_config['infer_mode'] = True
        self.ops = create_operators(self.transforms, self.global_config)
        self.model.eval()
    
    def predict(self, img_list):
        for img_data in img_list:   #get_image_file_list(img_path):
            try:
                _, encoded_image = cv2.imencode(".jpg", img_data)
            except:
                # print(len(img_data))
                # print(img_data)
                continue
            img = encoded_image.tobytes() 
            data = {'image': img}
            batch = transform(data, self.ops)
            if self.config['Architecture']['algorithm'] == "SRN":
                encoder_word_pos_list = np.expand_dims(batch[1], axis=0)
                gsrm_word_pos_list = np.expand_dims(batch[2], axis=0)
                gsrm_slf_attn_bias1_list = np.expand_dims(batch[3], axis=0)
                gsrm_slf_attn_bias2_list = np.expand_dims(batch[4], axis=0)

                others = [
                    paddle.to_tensor(encoder_word_pos_list),
                    paddle.to_tensor(gsrm_word_pos_list),
                    paddle.to_tensor(gsrm_slf_attn_bias1_list),
                    paddle.to_tensor(gsrm_slf_attn_bias2_list)
                ]
            if self.config['Architecture']['algorithm'] == "SAR":
                valid_ratio = np.expand_dims(batch[-1], axis=0)
                img_metas = [paddle.to_tensor(valid_ratio)]
            if self.config['Architecture']['algorithm'] == "RobustScanner":
                valid_ratio = np.expand_dims(batch[1], axis=0)
                word_positons = np.expand_dims(batch[2], axis=0)
                img_metas = [
                    paddle.to_tensor(valid_ratio),
                    paddle.to_tensor(word_positons),
                ]
            if self.config['Architecture']['algorithm'] == "CAN":
                image_mask = paddle.ones(
                    (np.expand_dims(
                        batch[0], axis=0).shape), dtype='float32')
                label = paddle.ones((1, 36), dtype='int64')
            images = np.expand_dims(batch[0], axis=0)
            images = paddle.to_tensor(images)
            if self.config['Architecture']['algorithm'] == "SRN":
                preds = self.model(images, others)
            elif self.config['Architecture']['algorithm'] == "SAR":
                preds = self.model(images, img_metas)
            elif self.config['Architecture']['algorithm'] == "RobustScanner":
                preds = self.model(images, img_metas)
            elif self.config['Architecture']['algorithm'] == "CAN":
                preds = self.model([images, image_mask, label])
            else:
                preds = self.model(images)
            post_result = self.post_process_class(preds)
            info = None
            if isinstance(post_result, dict):
                rec_info = dict()
                for key in post_result:
                    if len(post_result[key][0]) >= 2:
                        rec_info[key] = {
                            "label": post_result[key][0][0],
                            "score": float(post_result[key][0][1]),
                        }
                info = json.dumps(rec_info, ensure_ascii=False)
            elif isinstance(post_result, list) and isinstance(post_result[0],
                                                              int):
                # for RFLearning CNT branch 
                info = str(post_result[0])
            else:
                if len(post_result[0]) >= 2:
                    info = post_result[0][0] + "\t" + str(post_result[0][1])

            # if info is not None:
            #     print(info)
            yield info
import uuid   
class OCR_process(object):
    def __init__(self, config_dict):
        # 文本检测
        self.ocr_det = Detection(config_dict["ocr_det_config"])
        self.ocr_det.load_checkpoint()
        # # 水平方向文本识别
        # self.ocr_h_rec = OCR_rec(config_dict["ocr_h_rec_config"])
        # self.ocr_h_rec.load_checkpoint()
        # # 竖直方向文本识别
        # self.ocr_v_rec = OCR_rec(config_dict["ocr_v_rec_config"])
        # self.ocr_v_rec.load_checkpoint()        
        
        self.ocr_rec = OCR_rec(config_dict["ocr_rec_config"])
        self.ocr_rec.load_checkpoint() 
        self.debug_show = True
    
    
    
    def test_per_img(self, img):
        flag, crop_img_list, result_list = False, [], []
        # result_list [[container_id_1, iso_num_1], [, ]] 
        boxes = self.ocr_det.predict([img])
        if len(boxes) > 0:
            flag = True
            for box in boxes[0]:
                x1 = box[0][0]
                y1 = box[0][1]
                x2 = box[2][0]
                y2 = box[2][1]
                crop_img = img[y1 : y2, x1 : x2]
                if (y2 - y1)  > (x2 - x1):
                    crop_img = cv2.rotate(crop_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                crop_img_list.append(crop_img)
           
            info_stream = self.ocr_rec.predict(crop_img_list)
            for info in info_stream:          
                ocr_str, score_str = info.split("\t")
                result_list.append([ocr_str, score_str])
          
            if self.debug_show:
                
                import copy
                img_debug = copy.deepcopy(img) 
                if len(result_list) > 0:
                    for i, result in enumerate(result_list):
                        x1, y1 = boxes[0][i][0][0], boxes[0][i][0][1]
                        x2, y2 = boxes[0][i][2][0], boxes[0][i][2][1]
                        ocr_str = result[0]
                        score_str = result[1]
                        img_debug = cv2.rectangle(img_debug, (x1, y1), (x2, y2), (128, 128, 0), 2)
                        img_debug = cv2.putText(img_debug, ocr_str+" : "+score_str[:4], (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (128, 128, 0), 2)
                    cv2.imwrite(f"{uuid.uuid1()}.jpg", img_debug)

        return flag, result_list
    
    def sort_boxes(self, boxes):
    # 定义排序规则的函数
        def box_sort_key(box):
            return (box[0][1], box[0][0])  # 先按照y，再按照x
        sorted_boxes = sorted(boxes, key=box_sort_key)
        return sorted_boxes
    
    def clearbox(self, boxes):
        newbox = []
        for box in boxes:
            if len(box)==0:
                continue
            for i in box:
                newbox.append(i.tolist())
        return newbox
    
    def process_imgs(self, img_list, v):
        # img_list = [crop_images(img, v) for img in img_list]
        # 待补充逻辑：一组图片输入时取最终结果
        result_list = []
        result_item = ["", ""] # 箱号，iso号
        score_item = [0, 0]
        boxes = self.ocr_det.predict(img_list)      

        data1 = []
        data2 = []

        for i, i_boxes in enumerate(boxes):
            # 单张图片里的文本检测bboxes
            crop_img_list = []
            result_item = ["", ""] # 箱号，iso号
            score_item = [0, 0]            
            img = img_list[i]
            sortboxes = self.sort_boxes(i_boxes) 
            for box in sortboxes:

                x1 = box[0][0]
                y1 = box[0][1]
                x2 = box[2][0]
                y2 = box[2][1]
                crop_img = img[max(y1 - 10, 0):min(y2 + 10, img.shape[0]), max(x1 - 10, 0):min(x2 + 10, img.shape[1])]

                # if crop_img.shape[1] > 200:
                #     crop_img = cv2.resize(crop_img, (320, 32))

                if (y2 - y1)  > (x2 - x1):
                    crop_img = cv2.rotate(crop_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                crop_img_list.append(crop_img)
            
            
            info_stream = self.ocr_rec.predict(crop_img_list) 
            fflag =False
            fglag =False
            for info in info_stream:     
                # !!! 这个要增加ocr_tool中的筛选逻辑 目前只给了简单的置信度筛选     
                ocr_str, score_str = info.split("\t") 
                
                score = float(score_str)

                # 该过程判断三段式箱号 即 4+7这种结构
                if len(ocr_str)==4 and 'U' in ocr_str:
                    contain_alpht = ocr_str
                    fflag = True
                    continue
                if fflag:
                    fflag =False
                    # if ocr_str.isdigit():
                    ocr_str = contain_alpht + ocr_str

                if (len(ocr_str)==6 or len(ocr_str)==7) and ocr_str.isdigit():
                    fglag = True
                    digit_ocr = ocr_str
                    continue
                if fglag:
                    fglag=False
                    if(len(ocr_str)==4 and 'U' in ocr_str):
                        ocr_str = ocr_str + digit_ocr

                if not ocr_check.isPartOfContainerCode(ocr_str) and not ocr_check.check_95code(ocr_str):
                    continue
                    
                # 判断箱号以及格式
                if len(ocr_str) > 5 and score > (score_item[0]-0.1):
                    check_flag = ocr_check.check_Container_code(ocr_str)
                    if check_flag:
                        result_item[0] = ocr_str
                        score_item[0] = score
                        data1.append(ocr_str)
                # 判断iso
                if len(ocr_str) <= 4 and score > (score_item[1]-0.1) and ocr_check.check_95code(ocr_str):
                    result_item[1] = ocr_str
                    score_item[1] = score
                    data2.append(ocr_str)
            # result_list.append(result_item)
        # data1 = [data[0] for data in result_list]
        # data2 = [data[1] for data in result_list]
        
        final_result1 = getstr(data1)
        ocr_flag = ocr_check.check_Container_code(final_result1)
        if(ocr_flag=="0"):
            if(len(final_result1)==10):
                new_check = ocr_check.check_code_count(final_result1)
                final_result1 = final_result1 + new_check
            new_check = ocr_check.check_code_count(final_result1[:-1])
            print("right check mode:", new_check)
            final_result1 = final_result1[:-1] + new_check

        final_result2 = getstr(data2)

        if(len(final_result1)!=11):
            final_result1=''
        if(len(final_result2)!=4):
            final_result2=''

        # if len(data1) == 0:
        #     data1 = ''
        # else:
        #     data1 = data1[0]

        # if len(data2) == 0:
        #     data2 = ''
        # else:
        #     data2 = data2[0]
        # print('======================================')
        # print(final_result1, final_result2)
        # print('======================================')
        res = (final_result1, final_result2)
        
        return res


    def det_rec(self, imglist):
        info_stream = self.ocr_rec.predict(imglist)
        fflag =False
        result_list = []
        result_item = ["", ""] # 箱号，iso号
        score_item = [0, 0]
        data1 = []
        data2 = []
        for info in info_stream:     
            # !!! 这个要增加ocr_tool中的筛选逻辑 目前只给了简单的置信度筛选     
            ocr_str, score_str = info.split("\t") 
            if not ocr_check.isPartOfContainerCode(ocr_str) and not ocr_check.check_95code(ocr_str):
                continue
            # print(info.split("\t") , result_item, score_item)
            
            score = float(score_str)

            # 该过程判断三段式箱号 即 4+7这种结构
            if len(ocr_str)==4 and 'u' in ocr_str:
                contain_alpht = ocr_str
                fflag = True
                continue
            if fflag:
                fflag =False
                if ocr_str.isdigit():
                    ocr_str = contain_alpht + ocr_str
                
            # 判断箱号以及格式
            if len(ocr_str) > 5 and score > (score_item[0]-0.1):
                check_flag = ocr_check.check_Container_code(ocr_str)
                if check_flag:
                    result_item[0] = ocr_str
                    score_item[0] = score
                    data1.append(ocr_str)
            # 判断iso
            if len(ocr_str) <= 4 and score > (score_item[1]-0.1):
                result_item[1] = ocr_str
                score_item[1] = score
                data2.append(ocr_str)
        
        result_list.append(result_item)
        final_result1 = getstr(data1)
        ocr_flag = ocr_check.check_Container_code(final_result1)
        if(ocr_flag=="0"):
            if len(final_result1) == 10:
                new_check = ocr_check.check_code_count(final_result1)
                final_result1 = final_result1 + new_check
            new_check = ocr_check.check_code_count(final_result1[:-1])
            print("right check mode:", new_check)
            final_result1 = final_result1[:-1] + new_check
        final_result2 = getstr(data2)
        # print(data1, data2)
        # del img_list
        # del boxes
        # del info_stream
        if(len(final_result1)!=11):
            final_result1=" "
        if(len(final_result2)!=4):
            final_result2=" "
        return final_result1, final_result2

    
def getstr(strlist):
    transposed_data = list(zip_longest(*strlist, fillvalue=''))
    if(len(transposed_data)==0):
        return ''
    tp = transposed_data[-1]
    # print(tp)
    if('' in tp):
        # tp = transposed_data[-1]
        max_ele = Counter(tp).most_common(2) #对最后一行空值进行修改
        # print(max_ele)
        if '' not in max_ele[0][0]:
            elemm = max_ele[0][0]
        else:
            elemm = max_ele[1][0]
    else:
        elemm = Counter(tp).most_common(1)[0][0]
    transposed_data = transposed_data[:-1]
    # transposed_data = transposed_data.append(tp)
        # print(transposed_data)
    result = [Counter(column).most_common(1)[0][0] for column in transposed_data]
    result.append(elemm)
    # final_result1 = ''.join(result) 
    final_result1 = ''.join([str(item) for item in result])
    return final_result1

def vote2res(strlist):
    # [('BSIU3107448', '22G1'), ('BSIU3107448', '22G1')]
    # 初始化结果列表
    res1 = []
    res2 = []
    
    # 提取数据
    for ii in strlist:
        if ii[0] != '':
            res1.append(ii[0])
        if ii[1] != '':
            res2.append(ii[1])
    
    # 定义内部函数，用于获取结果
    def getres(res):
        # 转置数据
        transposed_data = list(zip_longest(*res, fillvalue=''))
        # 统计每列数据中出现最多次的元素
        result = [Counter(column).most_common(1)[0][0] for column in transposed_data]
        # 将结果拼接成字符串
        final_result = ''.join(result)
        return final_result
    
    # 获取结果
    res_con = getres(res1)
    res_iso = getres(res2)
    
    # 返回结果
    return res_con + res_iso


def crop_images(img, v):
    # v 表示当前图来源
    if v == 'right':
        # 相机的畸变参数
        distortion_coeffs = np.array([-0.39893605 ,-0.22947173 ,-0.00420022 , 0.02073844 , 0.96028294])
        # 相机的内参矩阵
        camera_matrix = np.array([[1617.65065868  ,  0. ,1037.21580999], [ 0. , 1620.74844035 ,729.50484112], [0, 0, 1]])
        undistorted_img = cv2.undistort(img, camera_matrix, distortion_coeffs)
        trapezoid = np.array([[893, 125], [2179, 337], [961, 1440], [2193, 857]], dtype=np.float32)
        rectangle = np.array([[0, 0], [640, 0], [0, 300], [640, 300]], dtype=np.float32)
        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(trapezoid, rectangle)
        # 进行透视变换
        result = cv2.warpPerspective(undistorted_img, M, (640,300))
        result = cv2.rotate(result, cv2.ROTATE_90_CLOCKWISE)

    elif v =='left':
        distortion_coeffs = np.array([-0.35520403 , 0.0737327  , 0.00719501 ,-0.01832554 , 0.02174101])
        # 相机的内参矩阵
        camera_matrix = np.array([[1390.971494 , 0., 1492.41532403], 
                                [ 0.      ,   1396.85961122 , 645.86962942], 
                                [0., 0., 1. ]])
        undistorted_img = cv2.undistort(img, camera_matrix, distortion_coeffs)
        trapezoid = np.array([[503, 525], [1891, 193], [539, 1029], [1967, 1440]], dtype=np.float32)
        rectangle = np.array([[0, 0], [640, 0], [0, 300], [640, 300]], dtype=np.float32)
        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(trapezoid, rectangle)
        # 进行透视变换
        result = cv2.warpPerspective(undistorted_img, M, (640, 300))
        result = cv2.rotate(result, cv2.ROTATE_90_COUNTERCLOCKWISE)
    

    elif v =='top':
        distortion_coeffs = np.array([-0.47161802, 0.12747005, -0.00210666, -0.00431911, 0.26938181])
        # 相机的内参矩阵
        camera_matrix = np.array([[1656.24104729, 0, 1309.77304108], [0, 1665.71015505, 725.67359664], [0, 0, 1]])
        # undistorted_img = cv2.undistort(img, camera_matrix, distortion_coeffs)
        undistorted_img = img
        trapezoid = np.array([[222, 178], [2264, 241], [168, 640], [2285, 724]], dtype=np.float32) # 左上、右上、左下、右下
        rectangle = np.array([[0, 0], [640, 0], [0, 200], [640, 200]], dtype=np.float32)
        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(trapezoid, rectangle)
        # 进行透视变换
        result = cv2.warpPerspective(undistorted_img, M, (640, 200))
        result = cv2.rotate(result, cv2.ROTATE_90_CLOCKWISE)
    else:
        result = img
    import time
    # cv2.imwrite(f'./test_/{time.time()}.jpg',result)
    return result



# 视频流测试部分
def run(video_file, rotate_tag):
    ocr_img_list = []

    config_dict = {
        "ocr_det_config": "./config/det/my_det_r50_db++_td_tr.yml",
        "ocr_rec_config": "./config/rec/my_en_PP-OCRv3_rec.yml"
        # "ocr_h_rec_config": "./config/rec/my_en_PP-OCRv3_rec.yml",
        # "ocr_v_rec_config": "./config/rec/my_en_PP-OCRv3_rec.yml"
    }
    my_ocr_process = OCR_process(config_dict)

    # 打开视频文件
    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        print("错误：无法打开视频文件。")
        exit()

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # img = crop_images(frame, "right")
            frame = cv2.resize(frame, (2560, 1440))
            ocr_img_list.append(frame)
            results = my_ocr_process.process_imgs(ocr_img_list, rotate_tag)
            ocr_img_list.clear()
            print(results)
            # print(frame.shape)
        else:
            break

    # 释放视频流对象和写入对象
    cap.release()
    


    



# from pympler import asizeof
# if __name__ == "__main__":
#     video = './test_/right.mp4'
#     rotate_tag = 'right'
#     run(video, rotate_tag)
    # # 参数文件传入

    # # tracemalloc.start()

    # config_dict = {
    #     "ocr_det_config": "./config/det/my_det_r50_db++_td_tr.yml",
    #     "ocr_rec_config": "./config/rec/my_en_PP-OCRv3_rec.yml"
    #     # "ocr_h_rec_config": "./config/rec/my_en_PP-OCRv3_rec.yml",
    #     # "ocr_v_rec_config": "./config/rec/my_en_PP-OCRv3_rec.yml"
    # }
    # my_ocr_process = OCR_process(config_dict)

    # vname = "/home/zhenjue4/xcy_work/tjGang/haveCarData/right.mp4"
    # cv =  cv2.VideoCapture(vname)
    # # 打开文件出错时报错
    # img_list = []
    # i=0
    # while cv.isOpened():  # 正常打开 开始处理
    #     i += 1
    #     rval, frame = cv.read()
    #     try:
    #         # img = cv2.resize(frame,(1920,1080))
    #         # new_img = cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE)
    #         # img = crop_images(frame,"right")
    #         # if(i<800 and i>400 and i%5==0):
    #         #     cv2.imwrite(f"./img/{uuid.uuid1()}.jpg", img)
    #         img_list.append(frame)
    #     except:
    #         break
        
    # cv.release()

    # results = my_ocr_process.process_imgs_2(img_list, 'right')
    # print(results)

    # img = cv2.imread("/home/hj1/sr/OCR/test/top.jpg") 
    # img_list.append(img)
    # flag, result = my_ocr_process.test_per_img(img)
    # if flag:
    #     print(result)

    # img = cv2.imread("/home/hj1/sr/OCR/test/ID_left1.jpg") 
    # img_list.append(img)
    # flag, result = my_ocr_process.test_per_img(img)
    # if flag:
    #     print(result)

    # img = cv2.imread("/home/hj1/sr/OCR/test/rear.jpg") 
    # img_list.append(img)
    # flag, result = my_ocr_process.test_per_img(img)
    # if flag:
    #     print(result)
    # process = psutil.Process()
    # imgtest = cv2.imread("/home/hj1/sr/OCR/what/frame_20231206163336597947.jpg")
    # my_ocr_process.test_per_img(imgtest)
    # for i in range(4):
    #     # memory_usage = asizeof.asizeof(my_ocr_process)/10**6
    #     # print(memory_usage)
    #     strlist = ['what', 'xz','new','test']
    #     img_list = []
    #     imgfile_path = "/home/zhenjue3/sr_work/ocr_data_extern/"+strlist[i]+"/"
    #     filelist = os.listdir(imgfile_path)
    #     for file in filelist:
    #         img_path = imgfile_path+file
    #         img = cv2.imread(img_path)
    #         # img = cv2.resize(img, (720,1280))
    #         # img = cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE)
    #         img_list.append(img)
        
    #     results = my_ocr_process.process_imgs(img_list)
    #     print(results)
        # del img_list
        # memory_usage = asizeof.asizeof(my_ocr_process)/10**6
        # print("aaa", memory_usage)
        # paddle.device.cuda.empty_cache()  
        # current, peak = tracemalloc.get_traced_memory()
        # print(f"out of Current memory usage: {current / 10**6} MB")
        # print(f"out of Peak memory usage: {peak / 10**6} MB")
        # print(results)
        # traceback_output = tracemalloc.get_object_traceback(my_ocr_process.process_imgs)

    # memory_info = process.memory_info()
    # print(memory_info)
    # tracemalloc.stop()

        

