import os
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import warnings
warnings.filterwarnings("ignore")
import tensorrt as trt
import threading
import time
import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from PIL import Image


class MobilenetTRT(object):
    def __init__(self, engine_file_path):
        # Create a Context on this device,
        self.ctx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        print("=======================")
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)
        print("===================")

        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        for binding in engine:
            print('binding:', binding, engine.get_binding_shape(binding))
            size = trt.volume(engine.get_binding_shape(
                binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                self.input_w = engine.get_binding_shape(binding)[-1]
                self.input_h = engine.get_binding_shape(binding)[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        self.batch_size = engine.max_batch_size

    def infer(self, raw_image_generator):
        """
        description: infer imgs of generator (bathsize = 1)
        """  
        threading.Thread.__init__(self)
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
        batch_image_raw = []
        batch_input_image = np.empty(
            shape=[self.batch_size, 3, self.input_h, self.input_w])
        for i, image_raw in enumerate(raw_image_generator):
            batch_image_raw.append(image_raw)
            input_image = self.pre_process(image_raw)
            # print(input_image)
            np.copyto(batch_input_image[i], input_image)
        batch_input_image = np.ascontiguousarray(batch_input_image)

        # Copy input image to host buffer
        np.copyto(host_inputs[0], batch_input_image.ravel())
        start = time.time()
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        # Run inference.
        context.execute_async(batch_size=self.batch_size,
                              bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        # Synchronize the stream
        stream.synchronize()
        end = time.time()
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
        # Here we use the first row of output in that batch_size = 1
        output = host_outputs[0]
        # Do postprocess
        return np.argmax(output), end - start

    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()

    def get_raw_image(self, img_list):
        """
        description: Read an image from image list
        """
        for img in img_list:
            yield img

    def get_raw_image_zeros(self, image_path_batch=None):
        """
        description: Ready data for warmup
        """
        for _ in range(self.batch_size):
            yield np.zeros([self.input_h, self.input_w, 3], dtype=np.uint8)

    def pre_process(self, img):
        """
        description: process img for model input 
        """
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img = img.resize((self.input_h, self.input_w))
        img = np.array(img)
        img = np.transpose(img, (2, 0, 1))
        img = img.astype(np.float32) / 255.0
        mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
        img = np.array((img - mean.reshape((3, 1, 1))) / std.reshape((3, 1, 1)))
        img = np.expand_dims(img, axis=0)   

        return np.ascontiguousarray(img)

class inferThread(threading.Thread):
    def __init__(self, mobilenet_wrapper, img_list):
        threading.Thread.__init__(self)
        self.mobilenet_wrapper = mobilenet_wrapper
        self.img_list = img_list
        self.result = None

    def run(self):
        result, use_time = self.mobilenet_wrapper.infer(
            self.mobilenet_wrapper.get_raw_image(self.img_list))
        #print('result->{}, time->{:.2f}ms'.format(
        #    result, use_time * 1000))
        self.result = result

    def get_result(self):
        return self.result

class warmUpThread(threading.Thread):
    def __init__(self, mobilenet_wrapper):
        threading.Thread.__init__(self)
        self.mobilenet_wrapper = mobilenet_wrapper
        self.result = None

    def run(self):
        result, use_time = self.mobilenet_wrapper.infer(
            self.mobilenet_wrapper.get_raw_image_zeros())
        print('result->{}, time->{:.2f}'.format(
            result, use_time * 1000))
        self.result = result

    def get_result(self):
        return self.result



if __name__ == '__main__':
    cfg_dict = {
        "engine_path": "./tianJinGang.engine",
        "class": [0, 1]  # 0 有车，result：1没车
    }
    mobilenet_wrapper = MobilenetTRT(cfg_dict["engine_path"])
    # root = './1'
    # file_lst = os.listdir(root)
    # for i in file_lst:
    #     img_pth = os.path.join(root, i)
    #     img = cv2.imread(img_pth)
    #     thread1 = inferThread(mobilenet_wrapper, [img])
    #     thread1.start()
    #     thread1.join()
    #     result = thread1.get_result()
    #     print(i, result)

    v_pth = r'D:\TianJinGangTest\25tup\top.mp4'
    decoder = cv2.cudacodec.createVideoReader(v_pth)
    count = 0
    while True:
        ret, frame = decoder.nextFrame()
        if not ret or frame is None:
            continue
        frame = frame.download()
        frame = frame[:, :, :3]
        seg_frame = frame[250:1350, 780:1880]
        thread1 = inferThread(mobilenet_wrapper, [seg_frame])
        thread1.start()
        thread1.join()
        result = thread1.get_result()
        if result == 0:
            count += 1
            print(f'====={count}======')
        else:
            count = 0

    # image = cv2.imread('./243.jpg')
    # image = cv2.imread('./243_gpu.jpg')
    # print(image.shape)
    # for i in range(500):
    #     t1 = time.time()
    #     thread1 = inferThread(mobilenet_wrapper, [image])
    #     thread1.start()
    #     thread1.join()
    #
    #     result = thread1.get_result()
    #     print(time.time() - t1)