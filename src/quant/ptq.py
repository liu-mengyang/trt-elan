import argparse
import yaml
import os
import math

import numpy as np

import torch
from torchvision import datasets, transforms
import torch.nn.functional as F
import tensorrt as trt
from cuda import cudart

from datas.utils import create_datasets


class CheckImage(object):
    
    def __init__(self):
        self.window_sizes = [4, 8, 16]
    
    def __call__(self, x):
        _, h, w = x.size()
        wsize = self.window_sizes[0]
        for i in range(1, len(self.window_sizes)):
            wsize = wsize*self.window_sizes[i] // math.gcd(wsize, self.window_sizes[i])
        mod_pad_h = (wsize - h % wsize) % wsize
        mod_pad_w = (wsize - w % wsize) % wsize
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

parser = argparse.ArgumentParser(description='ELAN')
parser.add_argument('--config', type=str, default=None, help='config file for evaluation and training')

args = parser.parse_args()
if args.config:
    opt = vars(args)
    yaml_args = yaml.load(open(args.config), Loader=yaml.FullLoader)
    opt.update(yaml_args)
print('use cuda & cudnn for acceleration!')
print('the gpu id is: {}'.format(args.gpu_ids))
device = torch.device('cuda')
torch.backends.cudnn.benchmark = True
torch.set_num_threads(args.threads)

# calibrate data
train_dataloaders, valid_dataloaders = create_datasets(args)

# calibrator
class MangaEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, val_data, cache_file, batch_size=32):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.cache_file = cache_file
        # Every time get_batch is called, the next batch of size batch_size will be copied to the device and returned.
        self.data = datasets.ImageFolder(val_data, transforms.Compose([
            transforms.ToTensor(),
            CheckImage()
        ]))
        self.dataset_length = len(self.data)
        self.batch_size = batch_size
        self.current_index = 0
        # Allocate enough memory for a whole batch.
        self.device_input = cudart.cudaMalloc(4 * 3 * 304 * 208 * self.batch_size)
    def get_batch_size(self):
        return self.batch_size
    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self, names):
        if self.current_index + self.batch_size > self.dataset_length:
            return None
        current_batch = int(self.current_index / self.batch_size)
        if current_batch % 10 == 0:
            print("Calibrating batch {:}, containing {:} images".format(current_batch, self.batch_size))
        batch = np.ascontiguousarray(torch.cat([self.data[i][0] for i in range(self.current_index, self.current_index + self.batch_size)], dim = 0).numpy().ravel())
        _, self.device_input = cudart.cudaMalloc(batch.nbytes)
        self.current_index += self.batch_size
        return [self.device_input]
    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
    def get_algorithm(self):
        return trt.CalibrationAlgoType.MINMAX_CALIBRATION

# original model data
class ModelData(object):
    MODEL_PATH = "../elan_x4.onnx"
    OUTPUT_NAME = "elan_x4_int8"
    # The original model is a float32 one.
    DTYPE = trt.float32

# build engine
TRT_LOGGER = trt.Logger()
def GiB(val):
    return val * 1 << 30

# This function builds an engine from a onnx model.
def build_int8_engine(onnx_filepath, calib, max_batch_size=32):
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    # trt.Runtime(TRT_LOGGER) as runtime
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(explicit_batch) as network, \
        builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser:
        # We set the builder batch size to be the same as the calibrator's, as we use the same batches
        # during inference. Note that this is not required in general, and inference batch size is
        # independent of calibration batch size.
        builder.max_batch_size = max_batch_size
        config.max_workspace_size = GiB(1) # 8G
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)
        config.int8_calibrator = calib
        # Parse model file
        with open(onnx_filepath, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    TRT_LOGGER.log(TRT_LOGGER.ERROR, parser.get_error(error))
                raise ValueError('Failed to parse the ONNX file.')
        TRT_LOGGER.log(TRT_LOGGER.INFO, f'input number: {network.num_inputs}')
        TRT_LOGGER.log(TRT_LOGGER.INFO, f'output number: {network.num_outputs}')
        # set optimization profile
        profile = builder.create_optimization_profile()
        input_name = network.get_input(0).name 
        profile.set_shape(input_name, min=(1, 3, 304, 208), opt=(1, 3, 304, 208), max=(1, 3, 304, 208))
        config.add_optimization_profile(profile)
        # Build engine and do int8 calibration.
        # 直接构造可以序列化的模型
#         plan = builder.build_serialized_network(network, config)
        # 反序列化
#         return runtime.deserialize_cuda_engine(plan)
        engine = builder.build_engine(network, config)
        with open('elan_x4_int8.plan', "wb") as f:
            f.write(engine.serialize())
        return engine

val_data = './calib_data'
calibration_cache = "manga_calibration.cache"
calib = MangaEntropyCalibrator(val_data, cache_file=calibration_cache, batch_size = 4)

# print(calib.data[0])

# Inference batch size can be different from calibration batch size.
batch_size = 256
onnx_file = ModelData.MODEL_PATH
engine = build_int8_engine(onnx_file, calib, batch_size)

# calibrate
# def load_test_case(pagelocked_buffer, img):
#     copy_size = img.ravel().size
#     np.copyto(pagelocked_buffer[:int(copy_size)], img.ravel())
# # Simple helper data class that's a little nicer to use than a 2-tuple.
# class HostDeviceMem(object):
#     def __init__(self, host_mem, device_mem):
#         self.host = host_mem
#         self.device = device_mem
#     def __str__(self):
#         return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)
#     def __repr__(self):
#         return self.__str__()
# # Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
# def allocate_buffers(engine):
#     inputs = []
#     outputs = []
#     bindings = []
#     stream = cuda.Stream()
#     print('max_batch_size', engine.max_batch_size)
#     for binding in engine:
#         print('binding', binding, engine.get_binding_shape(binding),engine.get_binding_dtype(binding))
#         size = trt.volume(engine.get_binding_shape(binding)[1:]) * engine.max_batch_size
#         print(size)
#         dtype = trt.nptype(engine.get_binding_dtype(binding))
#         # Allocate host and device buffers
#         host_mem = cuda.pagelocked_empty(size, dtype)
#         device_mem = cuda.mem_alloc(host_mem.nbytes)
#         # Append the device buffer to device bindings.
#         bindings.append(int(device_mem))
#         # Append to the appropriate list.
#         if engine.binding_is_input(binding):
#             inputs.append(HostDeviceMem(host_mem, device_mem))
#         else:
#             outputs.append(HostDeviceMem(host_mem, device_mem))
#     return inputs, outputs, bindings, stream

# inputs, outputs, bindings, stream = allocate_buffers(engine)