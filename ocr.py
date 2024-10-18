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

num_dic = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
p0_list = ['2', '4', '5', 'L']
p1_list = ['2', '5', 'C', 'F']
p2_list = ['G', 'R', 'U', 'P', 'T', 'V', 'K', '1']
p3_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'X', 'B']

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

def check_95code(code):
    # if code in iso6346.keys():
    #     return True
    # else:
    #     return False
    if len(code) != 4:
        return False
    if code[0] in p0_list and code[1] in p1_list and code[2] in p2_list and code[3] in p3_list:
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
    # print("sum_result", sum_result)
    if check_code == 10:  # 存在余数为10，校验码为1的情况
        return '0'
    else:
        return str(check_code)