# -*- coding:utf-8 -*-
import os
import sys
import cv2
from math import *
import numpy as np
from PIL import Image
from fuzzywuzzy import fuzz
import string
from fuzzywuzzy import process
import itertools
from itertools import combinations, permutations


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

import csv

inputCSV = "iso-container-codes.csv"
iso6346 = {
    "22K2": "罐式集装箱",
    "22K5": "罐式集装箱",
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
with open(os.path.join(os.path.dirname(__file__), owner_code_records)) as f:
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

fuzz.token_sort_ratio


def generate_owner_code_check_list():
    all_candidate_codes = set(permutations(string.ascii_uppercase, 1)).union(
        set(permutations(string.ascii_uppercase + string.ascii_uppercase, 2))).union(
        set(permutations(string.ascii_uppercase + string.ascii_uppercase + string.ascii_uppercase, 3)))
    all_candidate_codes = ["".join(s) for s in all_candidate_codes]
    container_head_check_dict = {}
    with open(os.path.join(os.path.dirname(__file__), "owner_code_check_list_temp.csv"), mode="w") as csvfile:
        for candidata in all_candidate_codes:
            container_head_check_dict[candidata] = process.extractBests(candidata,
                                                                        owner_code.keys(),
                                                                        score_cutoff=57,
                                                                        limit=100)
            print(candidata, container_head_check_dict[candidata])
            line = candidata
            for q in container_head_check_dict[candidata]:
                line = "{},{}".format(line, q[0])
            line = line + "\n"
            csvfile.write(line)


def read_owner_code_check_list():
    owner_code_check_list = {}
    path = "owner_code_check_list.csv"
    if not os.path.exists(path):
        return owner_code_check_list
    with open(os.path.join(os.path.dirname(__file__), path)) as csvfile:
        lines = csvfile.readlines()
        for line in lines:
            row = line.split(',')
            row[-1] = row[-1].strip()
            owner_code_check_list[row[0]] = row[1:]

    return owner_code_check_list


def generate_type_code_check_list():
    combine_list = [list(combinations([p0_list, p1_list, p2_list, p3_list], index)) for index in range(1, 5, 1)]
    all_candidate_codes = set()
    for combine_data in combine_list:
        for combine in combine_data:
            all_candidate_codes.update(["".join(s) for s in list(itertools.product(*combine))])

    type_code_check_dict = {}
    with open(os.path.join(os.path.dirname(__file__), "type_code_check_list.csv"), mode="w") as csvfile:
        for candidata in all_candidate_codes:
            type_code_check_dict[candidata] = process.extractBests(candidata,
                                                                   iso6346.keys(),
                                                                   scorer=fuzz.QRatio,
                                                                   score_cutoff=80,
                                                                   limit=100)
            if len(type_code_check_dict[candidata]) == 0:
                continue
            print(candidata, type_code_check_dict[candidata])

            line = candidata
            for q in type_code_check_dict[candidata]:
                line = "{},{}".format(line, q[0])
            line = line + "\n"
            csvfile.write(line)


if __name__ == '__main__':
    generate_type_code_check_list()
    pass
