import subprocess
import time
import sys

# # 检查是否提供了参数
# if len(sys.argv) < 2:
#     print("请提供 test_data 参数")
#     sys.exit(1)

test_data = "10191715"

# 获取参数值
# test_data = sys.argv[1]
# 启动第一个Python脚本
process1 = subprocess.Popen(['python', '计算资源数据.py'])
# 等待8秒
time.sleep(8)
# 启动第二个Python脚本，并传递参数
process2 = subprocess.Popen(['python', '0.1.0_alpha_102120000_top取流降帧_使用cpu读取数据.py', test_data])
# 等待两个进程结束
process1.wait()
process2.wait()

print("两个脚本都已运行结束。")
