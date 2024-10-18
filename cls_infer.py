# -*- coding: utf-8 -*-            
# @Author : Achen
# @Time : 2023/11/1 15:44


import os
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import warnings
warnings.filterwarnings("ignore")


from PIL import Image
import cv2
from torchvision.transforms import transforms
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
ENGINE_PATH = "./tianJinGang.engine"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def inference(image):
    runtime = trt.Runtime(TRT_LOGGER)
    assert runtime

    with open(ENGINE_PATH, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    assert engine

    context = engine.create_execution_context()
    assert context

    # 推理
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    image = transform(image)
    # 将图片转换为一个batch的输入
    data = image.unsqueeze(0)
    data = data.numpy()
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    inputs[0].host = data

    trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

    return np.argmax(trt_outputs)

import time
if __name__ == "__main__":
    image = cv2.imread('./243.jpg')
    for i in range(500):
        t1 = time.time()
    # result : 0 有车，result：1没车
        result = inference(image)
        print(time.time() - t1)
        print(result)
    