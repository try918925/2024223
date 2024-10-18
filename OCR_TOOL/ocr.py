# -*- coding:utf-8 -*-
import os
import sys
import cv2
# from math import *
from math import fabs, sin, cos, radians, degrees, atan2
import numpy as np

symbol_dic = {'A': 10, 'B': 12, 'C': 13, 'D': 14, 'E': 15,
              'F': 16, 'G': 17, 'H': 18, 'I': 19, 'J': 20,
              'K': 21, 'L': 23, 'M': 24, 'N': 25, 'O': 26,
              'P': 27, 'Q': 28, 'R': 29, 'S': 30, 'T': 31,
              'U': 32, 'V': 34, 'W': 35, 'X': 36, 'Y': 37, 'Z': 38}

num_dic = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
p0_list = ['2', '4', 'L']
p1_list = ['2', '5', 'C', 'F']
p2_list = ['G', 'R', 'U', 'P', 'T', 'V', 'K', '1']
p3_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'X', 'B']

p0_list_84 = ['2', '4', '9']
p1_list_84 = ['2', '5', '0', '3']
p2_list_84 = ['0', '3', '5', '7', '6', '8', '1']
p3_list_84 = ['0', '3', '4', '2', '1', '5']

inputCSV = "iso-container-codes.csv"
iso6346 = {
    "22K0": "罐式集装箱",
    "22K1": "罐式集装箱",
    "22K2": "罐式集装箱",
    "22K3": "罐式集装箱",
    "22K4": "罐式集装箱",
    "22K5": "罐式集装箱",
    "22K6": "罐式集装箱",
    "22K7": "罐式集装箱",
    "22K8": "罐式集装箱",
    "22KX": "罐式集装箱",
    "22RB": "",
    "22S1": "",
    "2CG1": "",
    "25G1": "",
    "25R1": "",
    "2EG1": "",
    "42GB": "",
    "42K7": "",
    "42KW": "",
    "42TG": "",
    "22GB": "",
    "45GB": "",
    "45U1": "",
    "LEG1": "",
    "221G": "",
    "2210": "",
    "20K2": "20201102新增",
    "4EG1": "",
    "28P2": "平台式容器",
    "12GB": "",
    "12RB": "",
    "25GB": "",
    "45RB": "",
    "25G2": "",
}
with open(os.path.join(os.path.dirname(__file__), inputCSV)) as f:
    for line in f:
        row = line.split(',')
        if row[0] == "code":
            continue
        iso6346[row[0]] = row[1]
owner_code_records = "owner_code_records.csv"
owner_code = {}
owner_code_guessing_dict_0 = {}
owner_code_guessing_dict_1 = {}
owner_code_guessing_dict_2 = {}
with open(os.path.join(os.path.dirname(__file__), owner_code_records), encoding='UTF-8') as f:
    for line in f:
        row = line.split(',')
        if len(row[0]) != 4 and row[0][-1] != "U":
            continue
        code = row[0]
        owner_code[code] = row[1:]
        sub_code_0 = code[1:3]
        sub_code_1 = code[0] + code[2:3]
        sub_code_2 = code[:2]
        if sub_code_0 in owner_code_guessing_dict_0.keys():
            owner_code_guessing_dict_0[sub_code_0].append(code)
        else:
            owner_code_guessing_dict_0[sub_code_0] = [code]

        if sub_code_1 in owner_code_guessing_dict_1.keys():
            owner_code_guessing_dict_1[sub_code_1].append(code)
        else:
            owner_code_guessing_dict_1[sub_code_1] = [code]

        if sub_code_2 in owner_code_guessing_dict_2.keys():
            owner_code_guessing_dict_2[sub_code_2].append(code)
        else:
            owner_code_guessing_dict_2[sub_code_2] = [code]

owner_code_check_list = {"": []}
with open(os.path.join(os.path.dirname(__file__), "owner_code_check_list.csv")) as csvfile:
    lines = csvfile.readlines()
    for line in lines:
        row = line.split(',')
        row[-1] = row[-1].strip()
        owner_code_check_list[row[0]] = row[1:]

type_code_check_list = {}
with open(os.path.join(os.path.dirname(__file__), "type_code_check_list.csv")) as csvfile:
    lines = csvfile.readlines()
    for line in lines:
        row = line.split(',')
        row[-1] = row[-1].strip()
        type_code_check_list[row[0]] = row[1:]


def sort_box(box):
    """
    对box进行排序
    """
    box = sorted(box, key=lambda x: sum([x[1], x[3], x[5], x[7]]))
    return box


def dumpRotateImage(img, degree, pt1, pt2, pt3, pt4):  # 进行旋转角度后图片提取
    height, width = img.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
    matRotation = cv2.getRotationMatrix2D((width // 2, height // 2), degree, 1)
    matRotation[0, 2] += (widthNew - width) // 2
    matRotation[1, 2] += (heightNew - height) // 2
    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    pt1 = list(pt1)
    pt3 = list(pt3)

    [[pt1[0]], [pt1[1]]] = np.dot(matRotation, np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(matRotation, np.array([[pt3[0]], [pt3[1]], [1]]))
    ydim, xdim = imgRotation.shape[:2]
    imgOut = imgRotation[max(1, int(pt1[1])): min(ydim - 1, int(pt3[1])),
             max(1, int(pt1[0])): min(xdim - 1, int(pt3[0]))]

    return imgOut


# 装载类型码９５码规则
def check_95code(code):
    if code in iso6346.keys():
        return True
    else:
        return False

    if len(code) != 4:
        return False
    if code[0] in p0_list and code[1] in p1_list and code[2] in p2_list and code[3] in p3_list:
        return True
    return False


# 装载类型码８４码规则
def check_84code(code):
    if len(code) != 4:
        return False
    if code[0] in p0_list_84 and code[1] in p1_list_84 and code[2] in p2_list_84 and code[3] in p3_list_84:
        return True
    return False


# 部分满足装载类型码９５码
def code_95_half(code):
    if len(code) >= 4:
        return False
    p_list = [p0_list, p1_list, p2_list, p3_list]

    if len(code) == 3:
        for i in range(2):
            if code[0] in p_list[i] and code[1] in p_list[1 + i] and code[2] in p_list[2 + i]:
                return True
    elif len(code) == 2:
        for i in range(3):
            if code[0] in p_list[i] and code[1] in p_list[1 + i]:
                return True
    elif len(code) == 1:
        for i in range(4):
            if code in p_list[i]:
                return True
    return False


# 　集装箱编号检验码验证
def check_Container_code(code_str):
    sum_result = 0
    check_flag = "0"  # 0表示失败，1表示成功
    sum_result = 0
    number = 0
    if len(code_str) != 11:  # 如果结果位数不为11，则结果本身不正确
        check_flag = "0"
    else:
        for i in range(len(code_str) - 1):  # 对结果每一位进行判断、计算
            code_content = code_str[i]
            if i < 4:  # 对前4位进行判断，应为字符。如果不为字符，则结果错误
                try:
                    code_num = symbol_dic[code_content]
                    sum_result = sum_result + code_num * num_dic[i]
                    number = number + 1
                except:
                    check_flag = "0"
                    break
            else:  # 对后面7位进行判断，如果不为数字，则结果错误
                try:
                    code_num = int(code_content)
                    sum_result = sum_result + code_num * num_dic[i]
                    number = number + 1
                except:
                    check_flag = "0"
                    break
    if sum_result > 0 and number == 10:  # 如果11位类型均正确，则对校验码进行判断
        check_code = sum_result % 11
        if str(check_code) == code_str[-1] or (check_code == 10 and code_str[-1] == '0'):  # 存在余数为10，校验码为1的情况
            check_flag = "1"
        else:
            check_flag = "0"

    return check_flag


# 投票方式确定编号（包括集装箱编号和装载类型编号）
def vote_code(code_list):
    ls = min(10, len(code_list[0]))
    nums = [{} for i in range(ls)]
    out_text = ''
    if len(code_list) <= 2:
        out_text = code_list[0][:ls]
    else:
        for name in code_list:
            for i in range(ls):
                if name[i] not in nums[i].keys():
                    nums[i][name[i]] = 1
                else:
                    nums[i][name[i]] += 1

        for num in nums:
            out = ''
            j_out = 0
            for i in num.keys():
                if j_out < num[i]:
                    j_out = num[i]
                    out = i
            out_text = out_text + out
    return str(out_text)


# 　补充集装箱编号的检验位
def check_code_count(code_str):
    sum_result = 0
    for i in range(len(code_str)):  # 对结果每一位进行判断、计算
        code_content = code_str[i]
        if i < 4:  # 对前4位进行判断，应为字符。如果不为字符，则结果错误
            try:
                code_num = symbol_dic[code_content]
                sum_result = sum_result + code_num * num_dic[i]
            except:
                return '0'

        else:  # 对后面7位进行判断，如果不为数字，则结果错误
            try:
                code_num = int(code_content)
                sum_result = sum_result + code_num * num_dic[i]
            except:
                return '0'

    check_code = sum_result % 11
    if check_code == 10:  # 存在余数为10，校验码为1的情况
        return '0'
    else:
        return str(check_code)


def Result_classification(results):
    '''
    针对OCR识别结果进行分类的函数。
    根据编号的规则及编号的形式进行分类，主要分为集装箱编号+装载类型编号两部分，
    其中装载类型主要是指满足95或84的装载码编号规则进行选择，
    而集装箱编号主要根据11位的具体组合方式（包括11,4+7,4+3+3+1三个方面）进行分类。
    '''
    str_num_11 = []
    num_7 = []
    num_n = []
    str_4 = []
    code_95 = []
    str_num_over11 = []
    str_num_less10 = []
    str_num_10 = []
    str_over4 = []
    str_less4 = []
    num_less7 = []
    num_over7 = []
    half_code95 = []
    special_code84 = []
    code_95_candidate = []

    for res in results:
        ocr_str = res[0]
        if ocr_str == '':
            continue
        ocr_str = str(ocr_str)
        if len(ocr_str) == 11 and ocr_str[3] == 'U':
            str_num_11.append(res)
        elif check_95code(ocr_str):
            code_95.append(res)
        elif len(ocr_str) == 6 and check_84code(ocr_str[-4:]) and ocr_str[:2].isalpha():
            special_code84.append(res)
        elif ocr_str.isdigit():
            if len(ocr_str) == 7:
                num_7.append(res)
            elif len(ocr_str) == 3 or len(ocr_str) == 1:
                num_n.append(res)
            elif len(ocr_str) < 7:
                num_less7.append(res)
            elif len(ocr_str) > 7:
                num_over7.append(res)
        elif ocr_str.isalpha():
            # print(ocr_str)
            if len(ocr_str) == 4 and ocr_str[-1] == 'U':
                str_4.append(res)
            elif len(ocr_str) < 4 and ocr_str[-1] == 'U':
                str_less4.append(res)
            elif len(ocr_str) > 4 and ocr_str[-1] == 'U':
                str_over4.append(res)
        elif len(ocr_str) > 11 and 'U' in ocr_str:
            str_num_over11.append(res)
        elif len(ocr_str) == 10 and ocr_str[:4].isalpha() and ocr_str[4:].isdigit():
            str_num_10.append(res)

        else:
            for num in range(4, len(ocr_str) + 1):
                if check_95code(ocr_str[num - 4:num]):
                    code_95_candidate.append([ocr_str[num - 4:num], res[1]])
                    break
            str_num_less10.append(res)

            if code_95_half(ocr_str):
                if len(half_code95) == 0:
                    half_code95.append(ocr_str)
                else:
                    if len(ocr_str) > len(half_code95[0]):
                        half_code95[0] = ocr_str

    if len(code_95) == 0:
        code_95 = code_95_candidate
    return str_num_11, num_7, num_n, str_4, code_95, str_num_over11, str_num_less10, str_num_10, str_less4, num_less7, num_over7, half_code95, str_over4, special_code84


# 针对22G1等形式的装载类型编号进行选择。(占时没用)
# 该部分选择方式是按照YOLO检测的text中心点与字符区域的中心点的距离为主要参照，选取距离最小并且置信度>4的第一个编号。
def choose_code_95(str_list):
    dis = [float(str_list[i][1]) for i in range(len(str_list))]
    sorted_dis = sorted(enumerate(dis), key=lambda x: x[1])
    output_str = ''
    for i in range(len(sorted_dis)):
        try:
            if float(str_list[sorted_dis[i][0]][2]) > 4.0:
                output_str = str_list[sorted_dis[i][0]][0]
                # code = str_list[sorted_dis[i][0]][2]
                break
        except:
            continue
    if output_str == '':
        output_str = str_list[sorted_dis[0][0]][0]
    return output_str


# 该部分选择方式是按照YOLO检测的text中心点与字符区域的中心点的距离为主要参照，选取距离最小的编号。
def choose_str_num(str_list):
    dis = [float(str_list[i][1]) for i in range(len(str_list))]
    sorted_dis = sorted(enumerate(dis), key=lambda x: x[1])
    output_str = str_list[sorted_dis[0][0]][0]
    return output_str


# 这部分指针对满足11位要求的编号部分进行选择，该部分选择方式是按照置信度最大的方式进行选择。
def choose_str11(str_list):
    output_str_num = ''
    goodCheckCode = []
    for i in range(len(str_list)):
        if check_Container_code(str_list[i][0]) == '1':
            goodCheckCode.append(str_list[i])

    if len(goodCheckCode) == 0:
        dis = [float(str_list[i][2]) for i in range(len(str_list))]
        sorted_dis = sorted(enumerate(dis), key=lambda x: x[1], reverse=True)
        output_str_num = str_list[sorted_dis[0][0]][0]
    else:
        dis = [float(goodCheckCode[i][2]) for i in range(len(goodCheckCode))]
        sorted_dis = sorted(enumerate(dis), key=lambda x: x[1], reverse=True)
        output_str_num = goodCheckCode[sorted_dis[0][0]][0]
    return output_str_num


# 针对大于11位的编号部分，提取出11位有效部位的选择，主要通过YOLO检测的text中心点与字符区域的中心点的距离最小为选择对象。
def choose_over11(str_list):
    output_str_num = ''
    output_ = ['' for i in range(len(str_list))]
    for i in range(len(str_list)):
        str_num = str_list[i][0]
        Lu = str_num.rindex('U')
        if Lu >= 3:
            output_[i] = str_num[Lu - 3:Lu + 1] + str_num[Lu + 1:min(Lu + 8, len(str_num))]
    dis = [float(str_list[i][1]) for i in range(len(str_list))]
    sorted_dis = sorted(enumerate(dis), key=lambda x: x[1])
    for i in range(len(sorted_dis)):
        if output_[sorted_dis[i][0]] != '' and len(output_[sorted_dis[i][0]]) == 11:
            output_str_num = output_[sorted_dis[i][0]]
            return output_str_num
    return output_str_num


# 选择编号中数字部分，依次添加，并且保证长度不大于7。
def choose_num_com(num_n):
    output_num = ''
    for res in num_n:
        if len(output_num) + len(res[0]) <= 7:
            output_num += res[0]
    return output_num


def TextCheck(results):
    output_num_7 = ''
    output_num = ''
    output_text = ''  # 输出
    output_str_num = ''
    output_IOS = ''

    str_num_11, num_7, _, str_4, code_95, str_num_over11, str_num_less10, str_num_10, str_less4, \
    num_less7, num_over7, half_code95, str_over4, special_code84 = Result_classification(results)  # 结果分类

    if len(str_4) > 0 and len(num_7) > 0:
        str_num_11.append(choose_str_num(str_4) + choose_str_num(num_7))

    if len(num_less7) > 0:
        str_num_less10 = str_num_less10 + num_less7

    # 箱型判断
    if len(code_95) > 0:
        output_IOS = choose_str_num(code_95)
    elif len(special_code84) > 0:
        output_IOS = choose_str_num(special_code84)

    # 11位+4位的情况
    if len(str_num_11) > 0:
        output_str_num = choose_str11(str_num_11)

    # # 4位+7位+4位的情况
    # elif len(str_4) > 0 and len(num_7) > 0:
    #     output_str = choose_str_num(str_4)
    #     output_num_7 = choose_str_num(num_7)
    #     output_str_num = output_str + output_num_7

    # 长度>11位时，从中间截取有效11位结果
    elif len(str_num_over11) > 0:
        output_str_num = choose_over11(str_num_over11)

    # 长度=10位时，直接输出
    elif len(str_num_10) > 0:
        output_str_num = choose_str_num(str_num_10)

    # 长度<10位时，选取全部
    elif len(str_num_less10) > 0:
        output_str_num = choose_str_num(str_num_less10)

    # # 字符=4，数字>7时
    # elif len(str_4) > 0 and len(num_over7) > 0:
    #     output_str = choose_str_num(str_4)
    #     output_num_7 = choose_str_num(num_over7)
    #     output_str_num = output_str + output_num_7

    # 字符>4，数字>7时
    elif len(str_over4) > 0 and len(num_over7) > 0:
        output_str = choose_str_num(str_over4)[-4:]
        output_num_7 = choose_str_num(num_over7)

        for i in range(7, len(output_num_7)):
            output_str_num = output_str + output_num_7[i - 7: i]
            if check_Container_code(output_str_num) == '1':
                break

    # 字符=4，数字<7时
    elif len(str_4) > 0 and len(num_less7) > 0:
        output_str = choose_str_num(str_4)
        output_num_7 = choose_str_num(num_less7)
        output_str_num = output_str + output_num_7

    # 字符<4，数字=7时
    elif len(str_less4) > 0 and len(num_7) > 0:
        output_str = choose_str_num(str_less4)
        output_num_7 = choose_str_num(num_7)
        output_str_num = output_str + output_num_7

    # 字符<4，数字<7时
    elif len(str_less4) > 0 and len(num_less7) > 0:
        output_str = choose_str_num(str_less4)
        output_num_7 = choose_str_num(num_less7)
        output_str_num = output_str + output_num_7

    # 将结果进行合并
    if len(output_IOS) != 0:
        output_text = output_str_num + ' ' + output_IOS
    else:
        output_text = output_str_num
        for i in half_code95:
            output_text = output_text + ' ' + i

    return output_text


def Text_check(boxs, results, Direction, partImgout):  # 编号后处理
    '''
    单张图片OCR的后处理:
    针对水平：
    1、首先针对box的y值进行高度方向从上到下的排序，以保证4+3+3+1的方式时，两个3的依次选择问题
    2、针对results的所有结果进行分类，具体分类方式参见函数Result_classification
    3、针对分类结果进行相关合并，包括装载类型选择，集装箱编号合并，具体方式参见Text_check函数
    针对垂直：
    1、针对results的所有结果进行分类（与水平相同），具体分类方式参见函数Result_classification
    2、针对分类结果进行相关合并，包括装载类型选择，集装箱编号合并，具体方式参见Text_check函数
    '''
    output_num_7 = ''
    output_num = ''
    output_text = ''  # 输出
    output_str_num = ''
    output_IOS = ''
    image_framed_c = partImgout.copy()
    if Direction == 0:  # 0表示水平
        try:
            # 针对boxs进行高度方向的排序
            boxs = np.array(boxs)
            boxs_arg = np.argsort(boxs[:, 1])
            results = np.array(results)
            results = results[boxs_arg]
        except:
            results = results

        # 将OCR的结果在原图上呈现(水平：识别内容从上往下依次排列)
        for i in range(len(results)):
            cv2.putText(image_framed_c, results[i][0], (100, i * 50 + 300), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 0, 0),
                        1)

        str_num_11, num_7, num_n, str_4, code_95, str_num_over11, str_num_less10, str_num_10, \
        str_less4, num_less7, num_over7, half_code95, str_over4, special_code84 = Result_classification(results)  # 结果分类

        # 箱型判断
        if len(code_95) > 0:
            output_IOS = choose_str_num(code_95)
        elif len(special_code84) > 0:
            output_IOS = choose_str_num(special_code84)

        # 11位的情况
        if len(str_num_11) > 0:
            output_str_num = choose_str11(str_num_11)

        # 4位+7位的情况
        elif len(str_4) > 0 and len(num_7) > 0:
            output_str = choose_str_num(str_4)
            output_num_7 = choose_str_num(num_7)
            output_str_num = output_str + output_num_7

        # 4位+3位+3位+1位的情况
        elif len(str_4) > 0 and len(num_n) > 0:
            output_str = choose_str_num(str_4)
            output_num = choose_num_com(num_n)
            output_str_num = output_str + output_num

        # 长度>11位时，从中间截取有效11位结果
        elif len(str_num_over11) > 0:
            output_str_num = choose_over11(str_num_over11)

        # 长度=10位时，直接输出
        elif len(str_num_10) > 0:
            output_str_num = choose_str_num(str_num_10)

        # 字符>4，数字>7时
        elif len(str_over4) > 0 and len(num_over7) > 0:
            output_str = choose_str_num(str_over4)[-4:]
            output_num_7 = choose_str_num(num_over7)

            for i in range(7, len(output_num_7)):
                output_str_num = output_str + output_num_7[i - 7:i]
                if check_Container_code(output_str_num) == '1':
                    break

        # 长度<10位时，选取全部
        elif len(str_num_less10) > 0:
            output_str_num = choose_str_num(str_num_less10)

        # 字符=4，数字<7时
        elif len(str_4) > 0 and len(num_less7) > 0:
            output_str = choose_str_num(str_4)
            output_num_7 = choose_str_num(num_less7)
            output_str_num = output_str + output_num_7

        # 字符<4，数字=7时
        elif len(str_less4) > 0 and len(num_7) > 0:
            output_str = choose_str_num(str_less4)
            output_num_7 = choose_str_num(num_7)
            output_str_num = output_str + output_num_7

        # 字符<4，数字<7时
        elif len(str_less4) > 0 and len(num_less7) > 0:
            output_str = choose_str_num(str_less4)
            output_num_7 = choose_str_num(num_less7)
            output_str_num = output_str + output_num_7

        # 字符<4，数字为3\1时
        elif len(str_less4) > 0 and len(num_n) > 0:
            output_str = choose_str_num(str_less4)
            output_num = choose_num_com(num_n)
            output_str_num = output_str + output_num

        # 将结果进行合并
        if len(output_IOS) != 0:
            output_text = output_str_num + ' ' + output_IOS
        else:
            output_text = output_str_num
            for i in half_code95:
                output_text = output_text + ' ' + i

    # 垂直方向
    else:

        str_num_11, num_7, _, str_4, code_95, str_num_over11, str_num_less10, str_num_10, str_less4, \
        num_less7, num_over7, half_code95, str_over4, special_code84 = Result_classification(results)  # 结果分类

        # 箱型判断
        if len(code_95) > 0:
            output_IOS = choose_str_num(code_95)
        elif len(special_code84) > 0:
            output_IOS = choose_str_num(special_code84)

        # 11位+4位的情况
        if len(str_num_11) > 0:
            output_str_num = choose_str11(str_num_11)

        # 4位+7位+4位的情况
        elif len(str_4) > 0 and len(num_7) > 0:
            output_str = choose_str_num(str_4)
            output_num_7 = choose_str_num(num_7)
            output_str_num = output_str + output_num_7

        # 长度>11位时，从中间截取有效11位结果
        elif len(str_num_over11) > 0:
            output_str_num = choose_over11(str_num_over11)

        # 长度=10位时，直接输出
        elif len(str_num_10) > 0:
            output_str_num = choose_str_num(str_num_10)

        # 长度<10位时，选取全部
        elif len(str_num_less10) > 0:
            output_str_num = choose_str_num(str_num_less10)

        # 字符=4，数字>7时
        elif len(str_4) > 0 and len(num_over7) > 0:
            output_str = choose_str_num(str_4)
            output_num_7 = choose_str_num(num_over7)
            output_str_num = output_str + output_num_7

        # 字符>4，数字>7时
        elif len(str_over4) > 0 and len(num_over7) > 0:
            output_str = choose_str_num(str_over4)[-4:]
            output_num_7 = choose_str_num(num_over7)

            for i in range(7, len(output_num_7)):
                output_str_num = output_str + output_num_7[i - 7: i]
                if check_Container_code(output_str_num) == '1':
                    break

        # 字符=4，数字<7时
        elif len(str_4) > 0 and len(num_less7) > 0:
            output_str = choose_str_num(str_4)
            output_num_7 = choose_str_num(num_less7)
            output_str_num = output_str + output_num_7

        # 字符<4，数字=7时
        elif len(str_less4) > 0 and len(num_7) > 0:
            output_str = choose_str_num(str_less4)
            output_num_7 = choose_str_num(num_7)
            output_str_num = output_str + output_num_7

        # 字符<4，数字<7时
        elif len(str_less4) > 0 and len(num_less7) > 0:
            output_str = choose_str_num(str_less4)
            output_num_7 = choose_str_num(num_less7)
            output_str_num = output_str + output_num_7

        # 将结果进行合并
        if len(output_IOS) != 0:
            output_text = output_str_num + ' ' + output_IOS
        else:
            output_text = output_str_num
            for i in half_code95:
                output_text = output_text + ' ' + i

    return output_text, image_framed_c


def charRec(img, text_recs, img2, Direction, crnn_right, xy_dis, adjust=False):
    """
    加载OCR模型，进行字符识别
    """
    results = [0 for i in range(len(text_recs))]
    boxs = [i for i in range(len(text_recs))]
    xDim, yDim = img.shape[1], img.shape[0]
    x, y = img2.shape[1], img2.shape[0]
    # print(x,xDim)
    res_img = {}
    res_distance = []
    partImgout = img2.copy()
    for index, rec in enumerate(text_recs):
        # 将ctpn获得的文本框区域在原始图片上进行确定（主要CTPN在计算时已经将图片进行放缩，因此需要进行调整）
        for i in range(0, 7, 2):
            rec[i] = int(rec[i] * float(x / xDim))
        for i in range(1, 8, 2):
            rec[i] = int(rec[i] * float(y / yDim))

        xlength = int((rec[6] - rec[0]) * 0.1)
        ylength = int((rec[7] - rec[1]) * 0.2)
        if adjust:
            pt1 = (max(1, rec[0] - xlength), max(1, rec[1] - ylength))
            pt2 = (rec[2], rec[3])
            pt3 = (min(rec[6] + xlength, x - 2), min(y - 2, rec[7] + ylength))
            pt4 = (rec[4], rec[5])
        else:
            pt1 = (max(1, rec[0]), max(1, rec[1]))
            pt2 = (rec[2], rec[3])
            pt3 = (min(rec[6], x - 2), min(y - 2, rec[7]))
            pt4 = (rec[4], rec[5])

        degree = degrees(atan2(pt2[1] - pt1[1], pt2[0] - pt1[0]))  # 图像倾斜角度，当前角度为０

        partImg = dumpRotateImage(img2, degree, pt1, pt2, pt3, pt4)  # 进行旋转角度后图片提取,

        # 确定ctpn获得的文本框与YOLO获得的文本框的中心点的距离
        xy_dis_rec = [float(pt3[0] + pt1[0]) / 2.0, float(pt3[1] + pt1[1]) / 2.0]
        distance = ((xy_dis[0] - xy_dis_rec[0]) ** 2.0 + (xy_dis[1] - xy_dis_rec[1]) ** 2.0) ** 0.5
        # print(distance)
        res_img[distance] = [partImg, rec]
        res_distance.append(distance)

    res_distance.sort()

    for index in range(len(res_distance)):
        partImg = res_img[res_distance[index]][0]
        rec = res_img[res_distance[index]][1]

        partImg1 = cv2.cvtColor(partImg, cv2.COLOR_BGR2RGB)
        text = crnn_predict(partImg1, crnn_right)  # 调用CRNN模型进行字符识别

        # 将OCR的结果在原图上呈现(垂直：识别内容在ctpn框的边上)
        if Direction == 1:
            cv2.putText(partImgout, text[0], (max(1, rec[0]), max(1, rec[1])), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0),
                        1)
        color = (0, 0, 255)

        cv2.line(partImgout, (int(rec[0]), int(rec[1])), (int(rec[2]), int(rec[3])), color, 2)
        cv2.line(partImgout, (int(rec[0]), int(rec[1])), (int(rec[4]), int(rec[5])), color, 2)
        cv2.line(partImgout, (int(rec[6]), int(rec[7])), (int(rec[2]), int(rec[3])), color, 2)
        cv2.line(partImgout, (int(rec[4]), int(rec[5])), (int(rec[6]), int(rec[7])), color, 2)

        if len(text[0]) > 0:
            boxs[index] = rec
            results[index] = [text[0], res_distance[index], text[1]]  # 数据存储，包括CRNN识别结果、与text中心的距离，识别结果的置信度
        else:
            boxs[index] = rec
            results[index] = ['', res_distance[index], 0.0]  # 识别文字

    output_text, partImgout = Text_check(boxs, results, Direction, partImgout)  # 调用图像OCR后处理

    return output_text, partImgout


def isPartOfContainerCode(code_str: str):
    if len(code_str) < 4:
        return False

    if len(code_str) == 4 and code_str[:4].isalpha() and code_str[3] == "U":
        return True

    if code_str[:4].isalpha() and code_str[4:].isdigit() and code_str[3] == "U":
        return True
    elif code_str[:3].isalpha() and code_str[3:].isdigit() and code_str[2] == "U":
        return True
    elif code_str[:2].isalpha() and code_str[2:].isdigit() and code_str[1] == "U":
        return True
    elif code_str[:1].isalpha() and code_str[1:].isdigit() and code_str[0] == "U":
        return True

    return False


def hasGoodContainerHead(code_str: str):
    if len(code_str) < 4:
        return False
    if len(code_str) == 4 and code_str[:4].isalpha() and code_str[3].upper() == "U":
        return True
    # print(code_str[:4].isalpha(), code_str[4:].isdigit(), code_str[3] == "U")
    if code_str[:4].isalpha() and code_str[4:].isdigit() and code_str[3].upper() == "U":
        return True


def hasGoodContainerBodys(code_str: str):
    if len(code_str) > 7 or len(code_str) <= 1:
        return False
    if code_str.isdigit():
        return True


def isGoodTypeCode(code_str: str):
    if check_95code(code_str):
        return True
    return False


isoLengthCode = {
    "1": "10",
    "2": "20",
    "3": "30",
    "4": "40",
    "B": "24",
    "C": "24′6″",
    "G": "41",
    "H": "43",
    "L": "45",
    "M": "48",
    "N": "49"
}
isoSecondSizeCode = {
    "0": "8′",
    "2": "8′6″",
    "4": "9′",
    "5": "9′6″",
    "6": "> 9′6″",
    "8": "4′3″",
    "9": "<= 4′",
    "C": "8′6″",
    "D": "9′",
    "E": "9′6″",
    "F": "> 9′6″"
}
isoThirdTpyeCode = {'G': "", "V": "", 'R': "", "H": "", 'U': "", 'P': "", "S": "", 'T': "", 'K': "", "B": ""}
isoFourthTpyeCode = {'0': "", '1': "", '2': "", '3': "", '4': "", '5': "", '6': "", '7': "", '8': "", '9': "", 'X': "",
                     'B': ""}
isoTypeCode = {
    "G0": "General - Openings at one or both ends",
    "G1": "General - Passive vents at upper part of cargo space",
    "G2": "General - Openings at one or both ends + full openings on one or both sides",
    "G3": "General - Openings at one or both ends + partial openings on one or both sides",
    "V0": "Fantainer - Non-mechanical, vents at lower and upper parts of cargo space",
    "V2": "Fantainer - Mechanical ventilation system located internally",
    "V4": "Fantainer - Mechanical ventilation system located externally",
    "R0": "Integral Reefer - Mechanically refrigerated",
    "R1": "Integral Reefer - Mechanically refrigerated and heated",
    "R2": "Integral Reefer - Self-powered mechanically refrigerated",
    "R3": "Integral Reefer - Self-powered mechanically refrigerated and heated",
    "H0": "Refrigerated or heated with removable equipment located externally; heat transfer coefficient K=0.4W/M2.K",
    "H1": "Refrigerated or heated with removable equipment located internally",
    "H2": "Refrigerated or heated with removable equipment located externally; heat transfer coefficient K=0.7W/M2.K",
    "H5": "Insulated - Heat transfer coefficient K=0.4W/M2.K",
    "H6": "Insulated - Heat transfer coefficient K=0.7W/M2.K",
    "U0": "Open Top - Openings at one or both ends",
    "U1": "Open Top - Idem + removable top members in end frames",
    "U2": "Open Top - Openings at one or both ends + openings at one or both sides",
    "U3": "Open Top - Idem + removable top members in end frames",
    "U4": "Open Top - Openings at one or both ends + partial on one and full at other side",
    "U5": "Open Top - Complete, fixed side and end walls ( no doors )",
    "T0": "Tank - Non-dangerous liquids, minimum pressure 0.45 bar",
    "T1": "Tank - Non-dangerous liquids, minimum pressure 1.50 bar",
    "T2": "Tank - Non-dangerous liquids, minimum pressure 2.65 bar",
    "T3": "Tank - Dangerous liquids, minimum pressure 1.50 bar",
    "T4": "Tank - Dangerous liquids, minimum pressure 2.65 bar",
    "T5": "Tank - Dangerous liquids, minimum pressure 4.00 bar",
    "T6": "Tank - Dangerous liquids, minimum pressure 6.00 bar",
    "T7": "Tank - Gases, minimum pressure 9.10 bar",
    "T8": "Tank - Gases, minimum pressure 22.00 bar",
    "T9": "Tank - Gases, minimum pressure to be decided",
    "B0": "Bulk - Closed",
    "B1": "Bulk - Airtight",
    "B3": "Bulk - Horizontal discharge, test pressure 1.50 bar",
    "B4": "Bulk - Horizontal discharge, test pressure 2.65 bar",
    "B5": "Bulk - Tipping discharge, test pressure 1.50 bar",
    "B6": "Bulk - Tipping discharge, test pressure 2.65 bar",
    "P0": "Flat or Bolster - Plain platform",
    "P1": "Flat or Bolster - Two complete and fixed ends",
    "P2": "Flat or Bolster - Fixed posts, either free-standing or with removable top member",
    "P3": "Flat or Bolster - Folding complete end structure",
    "P4": "Flat or Bolster - Folding posts, either free-standing or with removable top member",
    "P5": "Flat or Bolster - Open top, open ends (skeletal)",
    "S0": "Livestock carrier",
    "S1": "Automobile carrier",
    "S2": "Live fish carrier",
}


def isAGoodTypeCode(code_str: str):
    if len(code_str) == 6 and check_84code(code_str[-4:]) and code_str[:2].isalpha():
        return True

    if len(code_str) == 4:
        if code_str[0] in isoLengthCode.keys() and code_str[1] in isoSecondSizeCode.keys() and code_str[
            2] in isoThirdTpyeCode.keys() and code_str[3] in isoFourthTpyeCode.keys():
            return True

    if code_str in iso6346.keys():
        return True

    return False


def isAGoodOwnerCode(code_str: str):
    if code_str.isalpha() and len(code_str) == 4 and code_str[3] is "U":
        return True
    else:
        return False


def isAGoodContainerNumberPartCode(code_str: str):
    # 只看6位
    if code_str[:6].isdigit() and len(code_str[:6]) == 6:
        return True
    else:
        return False


def isACandidateTypeCode(code_str: str):
    if len(code_str) == 6 and check_84code(code_str[-4:]) and code_str[:2].isalpha():
        return True

    if len(code_str) > 4 and len(code_str) <= 7 \
            and (isAGoodTypeCode(code_str[:4]) or isACandidateTypeCode(code_str[:3])):
        return True

    if len(code_str) == 4:
        if code_str[0] in isoLengthCode.keys() and code_str[1] in isoSecondSizeCode.keys() and code_str[
            2] in isoThirdTpyeCode.keys() and code_str[3] in isoFourthTpyeCode.keys():
            return True

    elif len(code_str) == 3:
        if code_str[0] in isoLengthCode.keys() and code_str[1] in isoSecondSizeCode.keys() and code_str[
            2] in isoThirdTpyeCode.keys():
            return True
        if code_str[0] in isoSecondSizeCode.keys() and code_str[1] in isoThirdTpyeCode.keys() and code_str[
            2] in isoFourthTpyeCode.keys():
            return True
    elif len(code_str) == 2:
        if code_str[0] in isoLengthCode.keys() and code_str[1] in isoSecondSizeCode.keys():
            return True
        if code_str[0] in isoSecondSizeCode.keys() and code_str[1] in isoThirdTpyeCode.keys():
            return True
        if code_str[0] in isoThirdTpyeCode.keys() and code_str[1] in isoFourthTpyeCode.keys():
            return True
    else:
        return False


def isAgoodContainerCode(code_str: str):
    if check_Container_code(code_str) == "1":
        return True
    return False


def isACandidateContainerCode(code_str: str):
    # print("hasGoodContainerHead", hasGoodContainerHead(code_str))
    if (len(code_str) == 9 or len(code_str) == 10) and hasGoodContainerHead(code_str):
        return True
    return False


def isEndCheckCode(code_str: str):
    if len(code_str) == 1 and code_str.isdigit():
        return True
    return False


def textGenerate(results):
    def textCassification(results):

        containerHeads = []
        containerBodys = []
        containerEndCheckCodes = []

        goodTypeCodes = []
        candidateTypeCodes = []

        goodContainerCodes = []
        candidateContainerCodes = []

        endCheckCodes = []

        for result in results:
            ocr_str = result[0].upper()
            print("ocr_str", ocr_str)
            # 好的集装箱编号
            if isAgoodContainerCode(ocr_str):
                goodContainerCodes.append(result)

            # 10位可能的
            if isACandidateContainerCode(ocr_str):

                candidateContainerCodes.append(result)


            # 集装箱头
            if hasGoodContainerHead(ocr_str):
                containerHeads.append(result)

            # 中间的数字
            if hasGoodContainerBodys(ocr_str):
                containerBodys.append(result)

            if isEndCheckCode(ocr_str):
                containerEndCheckCodes.append(result)

            # 箱型码
            if isGoodTypeCode(ocr_str):
                goodTypeCodes.append(result)

            # 可能的箱形码
            if isACandidateTypeCode(ocr_str):
                candidateTypeCodes.append(result)
        return goodContainerCodes, goodTypeCodes, candidateContainerCodes, candidateTypeCodes, containerHeads, containerBodys, containerEndCheckCodes

    def getBestContainerCode(containerCodes: list):
        if len(containerCodes) == 0:
            return ""
        score = [float(containerCodes[i][2]) for i in range(len(containerCodes))]
        sorted_score = sorted(enumerate(score), key=lambda x: x[1], reverse=True)
        output_str_num = containerCodes[sorted_score[0][0]][0]
        return output_str_num

    def getBestCandidateContainerCode(containerCodes: list):
        if len(containerCodes) == 0:
            return ""
        temp = sorted(containerCodes, key=lambda x: len(x), reverse=True)
        for temp_container_code in temp:
            if isPartOfContainerCode(temp_container_code):
                return temp_container_code

        return temp[0]

    def generateCandidateContainerCode(containerHeads: list, containerBodys: list, containerEnds: list):

        def generateCollections(samples: list, limit: int):
            out = []
            if limit < 1:
                return out

            for index, sample in enumerate(samples):
                if len(sample) > limit:
                    continue
                restSamples = samples[:index] + samples[index + 1:]
                restLimit = limit - len(sample)
                if restLimit <= 0:
                    out.append(sample)
                else:
                    subCollection = generateCollections(restSamples, restLimit)
                    if len(subCollection) > 0:
                        for sub in subCollection:
                            out.append(sample + sub)
                    else:
                        out.append(sample)
            return out

        candidates = []

        for containerHead in containerHeads:
            headLen = len(containerHead)
            restNumberLen = 11 - headLen
            subCollection = generateCollections(containerBodys, restNumberLen)

            if len(subCollection) > 0:
                for sub in subCollection:
                    candidates.append(containerHead + sub)
            else:
                candidates.append(containerHead)

        for containerHead in containerHeads:
            headLen = len(containerHead)
            restNumberLen = 9 - headLen
            subCollection = generateCollections(containerBodys, restNumberLen)

            if len(subCollection) > 0:
                for sub in subCollection:
                    candidates.append(containerHead + sub)
            else:
                candidates.append(containerHead)

        for containerHead in containerHeads:
            headLen = len(containerHead)
            restNumberLen = 9 - headLen
            subCollection = generateCollections(containerBodys, restNumberLen)

            if len(subCollection) > 0:
                for sub in subCollection:
                    candidates.append(containerHead + sub)
            else:
                candidates.append(containerHead)

        for end in ends:
            for index, candidate in enumerate(candidates):
                if len(candidate) < 11:
                    candidates[index] = candidate + end

        return candidates

    def getBestTypeCode(typeCodes: list):
        if len(typeCodes) == 0:
            return ""
        score = [float(typeCodes[i][2]) for i in range(len(typeCodes))]
        sorted_score = sorted(enumerate(score), key=lambda x: x[1], reverse=True)
        output_str_num = typeCodes[sorted_score[0][0]][0]
        return output_str_num

    def getBestCandidateTypeCode(typeCodes: list):
        if len(typeCodes) == 0:
            return ""
        score = [float(typeCodes[i][2]) for i in range(len(typeCodes))]
        sorted_score = sorted(enumerate(score), key=lambda x: x[1], reverse=True)
        output_str_num = typeCodes[sorted_score[0][0]][0]
        return output_str_num

    # 数据分类
    goodContainerCodes, goodTypeCodes, candidateContainerCodes, candidateTypeCodes, containerHeads, containerBodys, containerEndCheckCodes = textCassification(
        results)
    containerCode = ""
    typeCode = ""
    print("textCassification:", textCassification(results))
    if len(goodContainerCodes) > 0:
        containerCode = getBestContainerCode(goodContainerCodes)
        # print("containerCode:", containerCode)
    else:
        heads = [containerHeads[i][0] for i in range(len(containerHeads))]
        bodys = [containerBodys[i][0] for i in range(len(containerBodys))]
        ends = [containerEndCheckCodes[i][0] for i in range(len(containerEndCheckCodes))]
        candidateContainerCodesEx = generateCandidateContainerCode(heads, bodys, ends)
        goodcandidates = []
        for candidate in candidateContainerCodesEx:
            if isAgoodContainerCode(candidate):
                goodcandidates.append(candidate)
        # print("goodcandidates: ", goodcandidates)
        # 筛选拼凑成的校验码过的
        containerCode = getBestCandidateContainerCode(goodcandidates)
        # print("containerCode: ", containerCode)
        # 筛选10位候补的
        if containerCode == "":
            candidateList = [candidateContainerCodes[i][0] for i in range(len(candidateContainerCodes))]
            containerCode = getBestCandidateContainerCode(candidateList)

        # 筛选所有的输入+ 拼凑的
        if containerCode == "":
            candidateList = [results[i][0] for i in range(len(results))]
            containerCode = getBestCandidateContainerCode(candidateList + candidateContainerCodesEx)

    if len(goodTypeCodes) > 0:
        typeCode = getBestTypeCode(goodTypeCodes)
    else:
        typeCode = getBestCandidateTypeCode(candidateTypeCodes)
        
    # print("containerCode", containerCode)
    return containerCode + " " + typeCode


def try_confusions(code: str):
    if not (len(code) == 11 and code[:4].isalpha() and code[4:9].isdigit()):
        return code, False
    alfa_confusion_dict = {
        "C": ["G", ],
        "D": ["O", ],
        "G": ["C", ],
        "H": ["M"],
        "I": ["T", ],
        "M": ["W", "H"],
        "N": ["M"],
        "O": ["D"],
        "R": ["P"],
        "T": ["I"],
        "W": ["M"]}
    digit_confusion_dict = {"0": ["8", ],
                            "5": ["8", "6"],
                            "6": ["8"],
                            "8": ["0"],
                            "9": ["8"]}
    for index, s in enumerate(code[:3]):
        if s in alfa_confusion_dict.keys():
            candicate_list = alfa_confusion_dict[s]
            for candicate in candicate_list:
                new_code = code[:index] + candicate + code[index + 1:]
                if new_code[:4] not in owner_code.keys():
                    continue
                if check_Container_code(new_code) == "1":
                    print("try_confusions OK!", code, "==>", new_code)
                    return new_code, True

    for index, s in enumerate(code[4:9]):
        if s in digit_confusion_dict.keys():
            candicate_list = digit_confusion_dict[s]
            for candicate in candicate_list:
                new_code = code[:index] + candicate + code[index + 1:]
                if check_Container_code(new_code) == "1":
                    print("try_confusions OK!", code, "==>", new_code)
                    return new_code, True

    return code, False


def split_container_code(full_code: str):
    index = len(full_code)
    for s in full_code[::-1]:
        if s.isalpha():
            break
        index -= 1
    if index == 0:
        container_head, container_number = "", full_code
    elif index == len(full_code):
        container_head, container_number = full_code, ""
    else:
        container_head, container_number = full_code[:index], full_code[index:]
    return container_head, container_number


def repair_container_code(code: str):
    # 原则，只要是没有通过校验码的串，都需要进行修复。
    # 分为修复头部字母串，中间数字串，尾部校验码三部分。
    # 首先进行头部字母串的检查与修复
    # 1.获得头部字母串
    # 规则,以最后出现的字母为分界,划分为字母串和数字串
    owner_code_error = 10
    container_number_error = 1
    check_number_error = -1
    error_info_dict = {owner_code_error: "owner code erorr", container_number_error: "number error",
                       owner_code_error + container_number_error: "owner code and number error",
                       owner_code_error + check_number_error: "owner code and check error",
                       check_number_error: "check error",
                       }

    def generate_confusion_digitals(digit):
        digit_confusion_dict = {"0": ["8", ],
                                "5": ["8", "6"],
                                "6": ["8", "5"],
                                "8": ["0", "6"],
                                "9": ["8"],
                                "1": ["7"],
                                "7": ["1"]}
        generate_list = []
        for index, s in enumerate(digit):
            if s in digit_confusion_dict.keys():
                candidate_list = digit_confusion_dict[s]
                for candidate in candidate_list:
                    new_code = digit[:index] + candidate + digit[index + 1:]
                    generate_list.append(new_code)
        return generate_list

    def generate_confusion_ownercodes(alfa):
        alfa_confusion_dict = {
            "C": ["G", ],
            "D": ["O", ],
            "G": ["C", ],
            "H": ["M"],
            "I": ["T", ],
            "M": ["W", "H"],
            "N": ["M"],
            "O": ["D"],
            "R": ["P"],
            "T": ["I"],
            "W": ["M"]}
        generate_list = []
        for index, s in enumerate(alfa):
            if s in alfa_confusion_dict.keys():
                candidate_list = alfa_confusion_dict[s]
                for candidate in candidate_list:
                    new_code = alfa[:index] + candidate + alfa[index + 1:]
                    generate_list.append(new_code)
        return generate_list

    if isAgoodContainerCode(code):
        return code, 1, "good code"

    origin_container_head, origin_container_number = split_container_code(code)

    if len(origin_container_number) < 6:
        return code, 0, error_info_dict[container_number_error]

    ##去掉中间的数字
    container_head = ''.join([i for i in origin_container_head if not i.isdigit()])

    # 修复分为有U和没有U
    # 有U则以U为分界进行修复,U前进行模糊匹配,U后部分进行删除
    container_head: str
    U_index = container_head.rfind("U")
    container_head = container_head[:U_index]

    if container_head not in owner_code_check_list.keys():
        return code, 0, error_info_dict[owner_code_error]

    q_container_head = owner_code_check_list[container_head]

    if U_index == 3:
        confusions_ownercodes = generate_confusion_ownercodes(container_head + "U")
    else:
        confusions_ownercodes = []

    # 模糊匹配也没有找到对应的head,那么就没有继续下去的必要了
    if len(q_container_head) == 0:
        return code, 0, error_info_dict[owner_code_error]

    # 2.检查现有head + 数字是否符合校验码
    if len(origin_container_number) >= 7:

        if len(origin_container_number) == 8:
            # 先判断是否存在子串是否正确
            stitch_code = q_container_head[0] + origin_container_number[:7]
            if isAgoodContainerCode(stitch_code):
                return stitch_code, 2, error_info_dict[container_number_error]

            stitch_code = q_container_head[0] + origin_container_number[-7:]
            if isAgoodContainerCode(stitch_code):
                return stitch_code, 2, error_info_dict[container_number_error]

        mid_six_number, check_number = origin_container_number[:6], origin_container_number[6]
        confusion_number_list = generate_confusion_digitals(mid_six_number)

        # 先枚举中间6位数字的混淆列表，每次只替换一位数字,头部不动
        # 如果通过校验码,说明数字串也可能有问题，包括校验码
        for confusion in confusion_number_list:
            stitch_code = q_container_head[0] + confusion + check_number
            if isAgoodContainerCode(stitch_code):
                return stitch_code, 2, error_info_dict[container_number_error]

        # 如果通过了校验码，认为是头部错了
        for head in confusions_ownercodes + q_container_head:
            stitch_code = head + origin_container_number
            if isAgoodContainerCode(stitch_code):
                return stitch_code, 2, error_info_dict[owner_code_error]

        # 将可能的head 与 混淆数字进行组合
        for head in q_container_head[:1] + confusions_ownercodes + q_container_head[1:]:
            for confusion in confusion_number_list:
                stitch_code = head + confusion + check_number
                if isAgoodContainerCode(stitch_code):
                    return stitch_code, 2, error_info_dict[owner_code_error + container_number_error]

        # 经过组合之后，仍没有合法的箱号，则只能认为是校验码错了，需要重新生成校验码
        best_head = q_container_head[0]
        stitch_code = best_head + mid_six_number
        if best_head == origin_container_head:
            error_type = check_number_error
        else:
            error_type = owner_code_error + check_number_error
        return checker(stitch_code), 2, error_info_dict[error_type]

    else:
        best_head = q_container_head[0]
        stitch_code = best_head + origin_container_number
        if best_head == origin_container_head:
            error_type = check_number_error
        else:
            error_type = owner_code_error + check_number_error
        return checker(stitch_code), 2, error_info_dict[error_type]


def repair_container_type_code(code: str, default="22G1"):
    if code in iso6346.keys():
        return code, 1, "good code"

    if code in type_code_check_list.keys():
        if len(code) == 3:
            if code + "1" in type_code_check_list[code]:
                # 最后是1的最常见
                return code + "1", 2, "error at tail"
            elif code[0] == "2" and "2" + code in type_code_check_list[code]:
                # 22的组合最常见
                return "2" + code, 2, "error at front"
            elif code[0] == "5" and "4" + code in type_code_check_list[code]:
                # 45的组合最常见
                return "4" + code, 2, "error at front"
            else:
                return type_code_check_list[code][0], 2, "unknown error"
        else:
            return type_code_check_list[code][0], 2, "unknown error"

    elif code[:4] in type_code_check_list.keys():
        return type_code_check_list[code[:4]][0], 2, "unknown error"
    elif code[:3] in type_code_check_list.keys():
        return type_code_check_list[code[:3]][0], 2, "unknown error"
    elif len(code) == 4:
        return code, 0, "unknown error"
    else:
        # 瞎猜就输出默认值 22G1
        return default, 0, "unknown error"


def fusion_container_code(code_list: str):
    if len(code_list) == 0:
        return "", 0, 0

    if len(code_list) == 1:
        # 如果只有一个，那就进行修复，修复和融合的侧重点不一样
        repaired_code, flag, info = repair_container_code(code_list[0])
        if flag == 1:
            score = 1
        elif flag == 2:
            score = 0.5
        else:
            score = 0
        return repaired_code, flag, score

    candidate_dict = {}

    good_score = 1
    good_special_score = good_score * 0.5
    almost_good_score = good_score * 0.5
    rapair_score = good_score * 0.4

    container_head_dict, container_number_dict = dict(), dict()
    good_list = []
    repair_list = []
    error_voted_list = []

    for code in code_list:
        if code not in candidate_dict.keys():
            candidate_dict[code] = 0
        container_head, container_number = split_container_code(code)
        container_number = container_number[:6]

        good_owner_code_flag = isAGoodOwnerCode(container_head)
        good_container_number_part_flag = isAGoodContainerNumberPartCode(container_number)

        if isAgoodContainerCode(code):
            # 好的编码,+1
            candidate_dict[code] += good_score
            good_list.append(code)
        elif len(code) == 11 and good_owner_code_flag and good_container_number_part_flag:
            # 校验码没过，但是在融合策略下不一定真错了。 + 1
            # 同时,修复校验码的也加分 但相对少加一点
            candidate_dict[code] += good_special_score
            error_voted_list.append(code)

            repaired_code = checker(code)
            if repaired_code not in candidate_dict.keys():
                candidate_dict[repaired_code] = 0
            candidate_dict[repaired_code] += 0.25

        elif good_owner_code_flag and good_container_number_part_flag:
            # 缺少校验码的, 补全后进行加分
            repaired_code = checker(code)
            repair_list.append(repaired_code)

            if repaired_code not in candidate_dict.keys():
                candidate_dict[repaired_code] = 0
            candidate_dict[repaired_code] += almost_good_score

        if good_owner_code_flag \
                and container_head not in container_head_dict.keys():
            container_head_dict[container_head] = 0

        if good_container_number_part_flag \
                and container_number not in container_number_dict.keys():
            container_number_dict[container_number] = 0

        if not (good_owner_code_flag and good_container_number_part_flag):
            # 非法字符,修复可能性不高,经过修复后+0.4
            repaired_code, flag, _ = repair_container_code(code)
            if repaired_code not in candidate_dict.keys():
                candidate_dict[repaired_code] = 0
            if flag != 0:
                candidate_dict[repaired_code] += rapair_score
                repair_list.append(repaired_code)

            # 获取可能的owner_code
            if good_owner_code_flag:
                container_head_dict[container_head] += 1

            # 获取可能的container_number_part
            if good_container_number_part_flag:
                container_number_dict[container_number] += 1

    if len(container_head_dict) > 0 and len(container_number_dict) > 0:
        # 可能存在组合的好结果
        best_head, best_head_score = sorted(container_head_dict.items(), key=lambda x: x[1], reverse=True)[0]
        best_container_number_part, best_container_number_part_score = \
            sorted(container_number_dict.items(), key=lambda x: x[1], reverse=True)[0]

        if best_head_score > 0 and best_container_number_part_score > 0:
            best_composed_code = checker(best_head + best_container_number_part)
            if best_composed_code not in candidate_dict.keys():
                candidate_dict[best_composed_code] = 0
            candidate_dict[best_composed_code] += almost_good_score
            repair_list.append(best_composed_code)

    fusion_result, fusion_score = sorted(candidate_dict.items(), key=lambda x: x[1], reverse=True)[0]
    if fusion_result in error_voted_list:
        for head in container_head_dict.keys():
            for container_number_part in container_number_dict.keys():
                candidate = checker(head + container_number_part)
                if candidate in good_list:
                    candidate_dict[candidate] += 1

        fusion_result, fusion_score = sorted(candidate_dict.items(), key=lambda x: x[1], reverse=True)[0]

    out_type = 0
    if fusion_result in good_list:
        out_type = 1
    elif fusion_result in error_voted_list:
        out_type = -1
    elif fusion_result in repair_list:
        out_type = 2
    else:
        pass
        # print("no good code!!!!")

    return fusion_result, out_type, fusion_score


def fusion_type_code(code_list: str):
    if len(code_list) == 1:
        # 如果只有一个，那就进行修复，修复和融合的侧重点不一样
        repaired_code, flag, info = repair_container_type_code(code_list[0])
        if flag == 1:
            score = 1
        elif flag == 2:
            score = 0.5
        else:
            score = 0
        return repaired_code, flag, score

    good_score = 1
    rapair_score = good_score * 0.4

    good_list = []
    repair_list = []
    error_voted_list = []
    candidate_dict = {}
    for code in code_list:
        if code not in candidate_dict.keys():
            candidate_dict[code] = 0
        if isAGoodTypeCode(code):
            candidate_dict[code] += good_score
            good_list.append(code)
        else:
            repaired_code, flag, info = repair_container_type_code(code)
            if repaired_code not in candidate_dict.keys():
                candidate_dict[repaired_code] = 0

            assert flag != 1
            if flag == 0:
                error_voted_list.append(repaired_code)
            else:
                repair_list.append(repaired_code)
                candidate_dict[repaired_code] += rapair_score

    fusion_result, fusion_score = sorted(candidate_dict.items(), key=lambda x: x[1], reverse=True)[0]
    out_type = 0
    if fusion_result in good_list:
        out_type = 1
    elif fusion_result in error_voted_list:
        out_type = -1
    elif fusion_result in repair_list:
        out_type = 2
    else:
        pass
        # print("no good type code!!!!")

    return fusion_result, out_type, fusion_score


def fusion_by_vote(container_info_list: list):
    '''
    测试用例：
    [[["EISU2007555", "22G1"]],
     [["EISU2007555", "22G1"], ["EISU2007555", "22G1"]],
     [["EISU2007555", "22G1"], ["EISU200755", "2G1"]],
     [["EISU2007555", "22G1"], ["EISU200755", "22G"], ["ISU2007555", "2G1"]],
     [["ISU2007555", "22G1"], ["EISU200755", "22G"], ["SU2007555", "2G1"]],
     [["ISU2007555", "22G1"], ["EISU200755", ""], ["ISU2007555", "2G1"]],
     [["EISU2007555", "22G1"], ["EISU200755", "22G"], ["ISU2607555", "22G"]]]
    :param container_info_list:
    :return:
    '''

    assert len(container_info_list) > 0
    # 确定候选名单a.原始的;b.自由组合的;c.经过修复的。
    container_code_list, type_code_list = [], []
    for container_info in container_info_list:
        container_code, type_code = container_info
        container_code_list.append(container_code)
        type_code_list.append(type_code)
    assert len(container_code_list) == len(type_code_list)

    fusion_result_container_code, container_code_flag, container_code_score = fusion_container_code(container_code_list)
    fusion_result_type_code, type_code_flag, type_code_score = fusion_type_code(type_code_list)
    return [fusion_result_container_code, fusion_result_type_code], \
           [container_code_flag, type_code_flag], \
           [container_code_score, type_code_score]


def is_container_code_matched(src, des):
    import Levenshtein

    if src == des:
        return True, 0

    src_owner_number, src_container_number = split_container_code(src)
    des_owner_number, des_container_number = split_container_code(des)
    MIN_MISS_MATCH_NUM = 2

    # 完全匹配
    if src_owner_number == des_owner_number and src_container_number == des_container_number:
        return True, 0

    owner_number_mis_match_num = Levenshtein.distance(src_owner_number, des_owner_number)
    mis_match_num = Levenshtein.distance(src_container_number, des_container_number)

    total_min_match_num = owner_number_mis_match_num + mis_match_num
    # 头匹配，数字不匹配
    if owner_number_mis_match_num == 0:
        # 数字串 错MIN_MISS_MATCH_NUM数量就算错
        if mis_match_num > MIN_MISS_MATCH_NUM:
            return False, total_min_match_num
        else:
            return True, total_min_match_num
    else:
        # 数字串 错MIN_MISS_MATCH_NUM -1 数量就算错
        if mis_match_num > MIN_MISS_MATCH_NUM - 1:
            return False, total_min_match_num
        else:
            return True, total_min_match_num


def is_type_code_matched(src, des):
    import Levenshtein

    if src == des:
        return True, 0

    MIN_MISS_MATCH_NUM = 2

    mis_match_num = Levenshtein.distance(src, des)
    total_min_match_num = mis_match_num
    # 头匹配，数字不匹配
    if mis_match_num == 0:
        # 数字串 错MIN_MISS_MATCH_NUM数量就算错
        if mis_match_num > MIN_MISS_MATCH_NUM:
            return False, total_min_match_num
        else:
            return True, total_min_match_num
    else:
        # 数字串 错MIN_MISS_MATCH_NUM -1 数量就算错
        if mis_match_num > MIN_MISS_MATCH_NUM - 1:
            return False, total_min_match_num
        else:
            return True, total_min_match_num


def is_container_code_matched_from_list(src, des_group: list):
    for des in des_group:
        match_flag, mis_match_num = is_container_code_matched(src, des)
        if match_flag is True:
            return True, mis_match_num

    return False, mis_match_num


def is_type_code_matched_from_list(src, des_group: list):
    for des in des_group:
        match_flag, mis_match_num = is_type_code_matched(src, des)
        if match_flag is True:
            return True, mis_match_num

    return False, mis_match_num


from functools import reduce
from itertools import chain


class SNChecker(object):
    # 以下一个字典和一个列表用于集装箱编号校验
    MAP_ALPHA_TO_NUM = {
        'A': 10, 'B': 12, 'C': 13, 'D': 14, 'E': 15, 'F': 16,
        'G': 17, 'H': 18, 'I': 19, 'J': 20, 'K': 21, 'L': 23,
        'M': 24, 'N': 25, 'O': 26, 'P': 27, 'Q': 28, 'R': 29,
        'S': 30, 'T': 31, 'U': 32, 'V': 34, 'W': 35, 'X': 36,
        'Y': 37, 'Z': 38
    }

    FACTORS_CHUNK1 = [1, 2, 4, 8]
    FACTORS_CHUNK2 = [16, 32, 64, 128, 256, 512]

    @classmethod
    def can_check(cls, serial_str):
        if len(serial_str) not in [10, 11]:
            return False
        # --------------------------------------------------
        if not serial_str[0:4].isalpha():
            return False
        if not serial_str[4:].isdecimal():
            return False
        # --------------------------------------------------
        return True

    def __call__(self, serial_str):
        code = self.calc_code(serial_str)
        if not code:
            return serial_str
        return serial_str[0:10] + code

    @classmethod
    def calc_code(cls, serial_str):
        if not cls.can_check(serial_str):
            return ''
        # --------------------------------------------------
        chunk1 = cls._convert_alpha_to_numbers(serial_str[0:4])
        chunk2 = cls._convert_numstr_to_numbers(serial_str[4:9])
        # --------------------------------------------------
        sum_result = cls._calc_sum(chunk1, chunk2)
        if not sum_result > 0:  # 如果计算和小于0
            return ''
        # ----------------------------------------
        code = sum_result % 11
        if code == 10:
            code = 0
        return str(code)

    @classmethod
    def _convert_alpha_to_numbers(cls, alpha_str):
        numbers = [cls.MAP_ALPHA_TO_NUM.get(item.upper())
                   for item in alpha_str]
        return numbers

    @classmethod
    def _convert_numstr_to_numbers(cls, num_str):
        numbers = [int(item) for item in num_str]
        return numbers

    @classmethod
    def _calc_sum(cls, chunk1, chunk2):
        return reduce(lambda x, y: x + y[0] * y[1],
                      chain(zip(chunk1, cls.FACTORS_CHUNK1),
                            zip(chunk2, cls.FACTORS_CHUNK2)), 0)


def checker(text):
    return SNChecker()(text)


if __name__ == '__main__':
    # test_list = ["EIBU2007555",
    #              "EISU2007555",
    #              "EISU2007551",
    #              "EISU200755",
    #              "ESSU2007555",
    #              "ESSU200755",
    #              "ESSU2001555",
    #              "ESSU200155"]
    # for test in test_list:
    #     print(test, repair_container_code(test))

    # test_list = ["22G1",
    #              "22G2",
    #              "22G",
    #              "452",
    #              "2G1",
    #              "251",
    #              "45",
    #              "5G1"]
    # for test in test_list[-1:]:
    #     print(test, repair_container_type_code(test))

    test_list = [[["EISU2007555", "22G1"]],
                 [["EISU2007555", "22G1"], ["EISU2007555", "22G1"]],
                 [["EISU2007555", "22G1"], ["EISU200755", "2G1"]],
                 [["EISU2007555", "22G1"], ["EISU200755", "22G"], ["ISU2007555", "2G1"]],
                 [["ISU2007555", "22G1"], ["EISU200755", "22G"], ["SU2007555", "2G1"]],
                 [["ISU2007555", "22G1"], ["EISU200755", ""], ["ISU2007555", "2G1"]],
                 [["EISU2007555", "22G1"], ["EISU200755", "22G"], ["ISU2607555", "22G"]]]
    test_list = [[["CCIU4110943",
                   ""], ["CCIU4110943",
                         "45G1"], ["COIU4110943",
                                   "45G1"], ["CCIU4110943",
                                             "45G1"]]]
    for index, test in enumerate(test_list[:]):
        print(fusion_by_vote(test))
    exit()
    test_list = ["BXTU2028909", "RXTU202890", "BXTU2028909", "RXTU2028909"]
    base = "BXTU2028909"
    for test in test_list[:]:
        # print(fusion_by_vote(test))
        pair = [base, test]

        print(pair, is_container_code_matched(*pair))

    pair = [base, test_list]
    print(pair, is_container_code_matched_from_list(*pair))
