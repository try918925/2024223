
import csv
import os

record_file_path = 'test.csv'






record_path = r'C:\Users\hello\Desktop\2024223\TianJinGangText\record_data'

a = "cpu:50.7% gpu:23% 显存:50% 内存:20.0%"
if "cpu" in a and "gpu" in a and "显存" in a and "内存" in a:
    b = a.replace(" ", ",").split(",")
    print(b)
    for i in b:
        if "cpu" in i:
            print(float(i.split(":")[1][:-1]))
        elif "显存" in i:
            print(float(i.split(":")[1][:-1]))
        elif "内存" in i:
            print(float(i.split(":")[1][:-1]))
        elif "gpu" in i:
            print(float(i.split(":")[1][:-1]))


with open(os.path.join(record_path, record_file_path), 'w', encoding='utf-8', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerows([['time', 'cpu', 'gpu', '显存', '内存']])
    csv_writer.writerows([['time', 'cpu', 'gpu', '显存', '内存']])
    csv_writer.writerows([['time', 'cpu', 'gpu', '显存', '内存']])
    csv_writer.writerows([['time', 'cpu', 'gpu', '显存', '内存']])