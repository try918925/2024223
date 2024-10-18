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
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
import paddle
import subprocess
import tracemalloc
from ppocr_car.data import create_operators, transform
from ppocr_car.modeling.architectures import build_model
from ppocr_car.postprocess import build_post_process
from ppocr_car.utils.save_load import load_model
from ppocr_car.utils.utility import get_image_file_list
# import tools.program as program
import cv2
import yaml
from collections import Counter
from itertools import zip_longest

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
        post_result, score = self.post_process_class(preds, shape_list)
        boxes = post_result[0]['points']
        return boxes, score
            
            
    def predict(self, img_list):
        img_box = []
        box_scores = []
        for img_data in img_list:
            h,w = img_data.shape[:2]
            _, encoded_image = cv2.imencode(".jpg", img_data)
            img = encoded_image.tobytes()
            box, score = self._predict2box(img)
            img_box.append(box)
            box_scores.append(score)
        return img_box, box_scores

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

    def process_imgs(self, img_list):
        # 待补充逻辑：一组图片输入时取最终结果
        result_list = []
        boxes, scores = self.ocr_det.predict(img_list[:50])
        for i, i_boxes in enumerate(boxes):
            # 单张图片里的文本检测bboxes
            crop_img_list = []
            img = img_list[i]
            score_det = scores[i]
            # sortboxes = self.sort_boxes(i_boxes) 
            # print(sortboxes)
            for box in i_boxes:
                x1 = box[0][0]
                y1 = box[0][1]
                x2 = box[2][0]
                y2 = box[2][1]
                # crop_img = img[y1: y2, x1: x2]
                # if(x1<279 or y1<528 or x2>800 or y2 > 800):
                #     return []
                crop_img = img[max(y1, 0):min(y2, img.shape[0]), max(x1, 0):min(x2, img.shape[1])]
                # h,w = crop_img.shape[:2]
                # crop_img = img[max(y1 - 10, 0):min(y2 + 10, img.shape[0]), max(x1 - 10, 0):min(x2 + 10, img.shape[1])]
                # if crop_img.shape[0] != 0 and crop_img.shape[1] != 0:
                #     crop_img = cv2.resize(crop_img, (w*3,h*3))
                # if (y2 - y1)  > (x2 - x1):
                #     crop_img = cv2.rotate(crop_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                crop_img_list.append(crop_img)
            
            info_stream = self.ocr_rec.predict(crop_img_list)
            # fflag =False
            for info in info_stream:     
                ocr_str, score_str = info.split("\t")
                score = float(score_str)
                if(score < 0.5 or len(ocr_str) != 7):
                    continue
                else:
                    if ocr_str[1].isalpha() and len(ocr_str) == 7:
                        result_list.append([ocr_str, score_det])
            
            # final_result = getstr(result_list)
        if len(result_list) > 0:
            return result_list[0]
        else:
            return result_list
    
def getstr(strlist):
    transposed_data = list(zip_longest(*strlist, fillvalue=''))
    if(len(transposed_data)==0):
        return ''
    tp = transposed_data[-1]
    if('' in tp):
        # tp = transposed_data[-1]
        max_ele = Counter(tp).most_common(2) #对最后一行空值进行修改
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
    final_result1 = ''.join(result) 
    return final_result1


def selectSave(threshLst):
    if len(threshLst) > 0:
        max_value = max(threshLst)
        max_index = threshLst.index(max_value)
        return max_index
    else:
        return 0
    

def get_finalResult(result_lst):
    tmp_thresh_lst = []
    tmp_result = []
    idx = 0
    for result, frame in result_lst:
        if len(result) > 1:
            if len(result[0]) == 7:
                tmp_result.append(result[0])
            tmp_thresh_lst.append(result[1])
        else:
            tmp_thresh_lst.append(0)
    if len(tmp_thresh_lst) > 0:
        idx = tmp_thresh_lst.index(max(tmp_thresh_lst))
    if(len(result_lst)>3):
        tmp_result_list = sorted(result_lst, key=lambda x: x[0][1], reverse=True)[:3]
        tmp_result = [item[0][0] for item in tmp_result_list]
    final_result = getstr(tmp_result)

    return final_result, idx










# from pympler import asizeof
if __name__ == "__main__":
    # 参数文件传入

    # tracemalloc.start()

    config_dict = {
        "ocr_det_config": "./config_car/det/my_car_det_r50_db++_td_tr.yml",
        "ocr_rec_config": "./config_car/rec/my_rec_chinese_lite_train_v2.0.yml"
        # "ocr_h_rec_config": "./config/rec/my_en_PP-OCRv3_rec.yml",
        # "ocr_v_rec_config": "./config/rec/my_en_PP-OCRv3_rec.yml"
    }
    my_ocr_process = OCR_process(config_dict)

    img_list_path = [r'C:\Users\Install\Desktop\2024223\result\01\240430\CAR0267\Transfer\chepai.jpg']
    
    result_lst = []
    save_frame = []
    thres_lst = []
    for i in range(len(img_list_path)):
        img = cv2.imread(img_list_path[i])
        useImg = img[500:1440, 600:1700]
        useImg = cv2.GaussianBlur(useImg, (5, 5), 0)
        result = my_ocr_process.process_imgs([useImg])
        print(result)
        result_lst.append(result[0])
        save_frame.append(img)
        thres_lst.append(result[1])

    # print(thres_lst)
    # saveIdx = selectSave(thres_lst)
    # print(getstr(result_lst))

    
    

    # for ip in img_list_path:
    #     img_list =[]
    #     img = cv2.imread(ip)
    #     img_list.append(img)
    #     result = my_ocr_process.process_imgs(img_list)
    #     print(f'车牌：{result[0]}', f'检测框置信度：{result[1][0]}')

    

