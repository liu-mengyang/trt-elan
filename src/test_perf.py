import argparse
import yaml
import time

import torch
import torch.nn as nn
import numpy as np

import onnx
import onnxruntime as ort

import tensorrt as trt
from cuda import cudart

import utils

tableHead = \
"""
bs: Batch Size
width: Image width
height: Image height
lt: Latency (ms)
tp: throughput (image/s)
max-a0: maximum of absolute difference of output 0
med-a0: median of absolute difference of output 0
mea-a0: mean of absolute difference of output 0
max-r0: maximum of absolute difference of output 0
med-r0: median of relative difference of output 0
mea-r0: mean of relative difference of output 0
----+-----+------+--------+---------+---------+---------+---------+---------+---------+---------+-------------
  bs|width|height|      lt|       tp|   max-a0|   med-a0|   mea-a0|   max-r0|   med-r0|   mea-r0| output check
----+-----+------+--------+---------+---------+---------+---------+---------+---------+---------+-------------
"""

def printArrayInfo(x, description=""):
    print( '%s: %s\n  Mean=%.5e,SumAbs=%.5e,Var=%.5e,Max=%.5f,Min=%.5f,SAD=%.5e'%( \
        description,str(x.shape),np.mean(x),np.sum(abs(x)),np.var(x),np.max(x),np.min(x),np.sum(np.abs(np.diff(x.reshape(-1)))) ))
    print("\t", x.reshape(-1)[:10])

def check(a, b, weak=False, epsilon = 1e-5):
    if weak:
        res = np.all( np.abs(a - b) < epsilon )
    else:
        res = np.all( a == b )
    abs_diff = np.abs(a - b)
    rel_diff = np.abs(a - b) / (np.abs(b) + epsilon)
    diff0 = np.max(abs_diff)
    diff1 = np.median(abs_diff)
    diff2 = np.mean(abs_diff)
    diff3 = np.max(rel_diff)
    diff4 = np.median(rel_diff)
    diff5 = np.mean(rel_diff)
    #print("check:",res,diff0,diff1)
    return res,diff0,diff1,diff2,diff3,diff4,diff5

parser = argparse.ArgumentParser(description='ELAN')
parser.add_argument('--config', type=str, default=None, help='config file for evaluation and training')

# prepare test data
batch_size = 1
batchSize = 1
channel = 3
height = 304
width = 208

test_lr = torch.randn([batch_size, channel, height, width], dtype=torch.float32)
print('****  ', batch_size, '  ****')

# pytorch baseline
print('====', 'pytorch', '====')

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
torch.set_grad_enabled(False)

model_raw = utils.import_module('models.{}_network'.format(args.model, args.model)).create_model(args)
model = nn.DataParallel(model_raw).to(device)
# Loading weights
print('load pretrained model: {}!'.format(args.pretrain))
ckpt = torch.load(args.pretrain)
model.load_state_dict(ckpt['model_state_dict'])
model = model.eval()

test_lr = model_raw.check_image_size(test_lr)
print(test_lr.shape)
test_lr_cu = test_lr.cuda()
nWarmRound = 10
for i in range(nWarmRound):
    output_pytorch = model(test_lr_cu)
print(output_pytorch.shape)

nRound = 100
torch.cuda.synchronize()
t0 = time.time()
for i in range(nRound):
    output_pytorch = model(test_lr_cu)
torch.cuda.synchronize()
time_pytorch = (time.time() - t0) * 1000 / nRound
print('Pytorch time: ', time_pytorch)
throughout_pytorch = 1000 / time_pytorch * batch_size
print('Pytorch throughout: ', throughout_pytorch)

# test onnx performance
print('====', 'ONNX', '====')
onnx_model = onnx.load("elan_x4.onnx")
onnx.checker.check_model(onnx_model)

ort_sess = ort.InferenceSession('elan_x4.onnx', providers=['CUDAExecutionProvider'])

nWarmRound = 10
for i in range(nWarmRound):
    output_onnx = ort_sess.run(None, {'lr': test_lr.numpy()})

nRound = 100
torch.cuda.synchronize()
t0 = time.time()
for i in range(nRound):
    output_onnx = ort_sess.run(None, {'lr': test_lr.numpy()})
torch.cuda.synchronize()
time_onnx = (time.time() - t0) * 1000 / nRound
# print('ONNX time: ', time_onnx)
# throughout_onnx = 1000 / time_onnx * batch_size
# print('ONNX throughout: ', throughout_onnx)

print('****Average diff between onnx and pytorch****')
print(tableHead)
timePerInference = time_onnx
check0 = check(output_onnx[0],output_pytorch.detach().cpu().numpy(),True,5e-5)
string = "%4d,%6d,%6d,%8.3f,%9.3e,%9.3e,%9.3e,%9.3e,%9.3e,%9.3e,%9.3e"%(batchSize,
                                                                        width,
                                                                        height,
                                                                        timePerInference,
                                                                        batchSize/timePerInference*1000,
                                                                        check0[1],
                                                                        check0[2],
                                                                        check0[3],
                                                                        check0[4],
                                                                        check0[5],
                                                                        check0[6])
print(string)
# print(string + ", %s"%("Good" if check0[1] < 3.5e-2 and check0[2] < 2e-3 and check1[2] < 1e-1 else "Bad"))

# test surgeoned onnx performance
print('====', 'ONNX surgeoned', '====')
onnx_sed_model = onnx.load("elan_x4_sed.onnx")
onnx.checker.check_model(onnx_sed_model)

ort_sess_sed = ort.InferenceSession('elan_x4_sed.onnx', providers=['CUDAExecutionProvider'])

nWarmRound = 10
for i in range(nWarmRound):
    output_onnx_sed = ort_sess_sed.run(None, {'lr': test_lr.numpy()})

nRound = 100
torch.cuda.synchronize()
t0 = time.time()
for i in range(nRound):
    output_onnx_sed = ort_sess_sed.run(None, {'lr': test_lr.numpy()})
torch.cuda.synchronize()
time_onnx_sed = (time.time() - t0) * 1000 / nRound
print('ONNX surgeoned time: ', time_onnx_sed)
# throughout_onnx_sed = 1000 / time_onnx_sed * batch_size
# print('ONNX surgeoned throughout: ', throughout_onnx_sed)

print('****Average diff between onnx_sed and pytorch****')
print(tableHead)
timePerInference = time_onnx_sed
check0 = check(output_onnx_sed[0],output_pytorch.detach().cpu().numpy(),True,5e-5)
string = "%4d,%6d,%6d,%8.3f,%9.3e,%9.3e,%9.3e,%9.3e,%9.3e,%9.3e,%9.3e"%(batchSize,
                                                                        width,
                                                                        height,
                                                                        timePerInference,
                                                                        batchSize/timePerInference*1000,
                                                                        check0[1],
                                                                        check0[2],
                                                                        check0[3],
                                                                        check0[4],
                                                                        check0[5],
                                                                        check0[6])
print(string)
# print(string + ", %s"%("Good" if check0[1] < 3.5e-2 and check0[2] < 2e-3 and check1[2] < 1e-1 else "Bad"))

# test FP32 tensorrt performance
print('====', 'TensorRT fp32', '====')
logger = trt.Logger(trt.Logger.ERROR)
with open('elan_x4.plan', 'rb') as f:
    engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
if engine == None:
    print("Failed loading engine!")
    exit()
print("Succeeded loading engine!")

context = engine.create_execution_context()
context.set_binding_shape(0, [1, 3, 304, 208])
print("EngineBinding0->", engine.get_binding_shape(0), engine.get_binding_dtype(0))
print("EngineBinding1->", engine.get_binding_shape(1), engine.get_binding_dtype(1))

inputH0 = np.ascontiguousarray(test_lr.numpy().reshape(-1))
outputH0 = np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1)))
_, inputD0 = cudart.cudaMalloc(inputH0.nbytes)
_, outputD0 = cudart.cudaMalloc(outputH0.nbytes)

nWarmRound = 10
for i in range(nWarmRound):
    cudart.cudaMemcpy(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    context.execute_v2([int(inputD0), int(outputD0)])
    cudart.cudaMemcpy(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

nRound = 100
torch.cuda.synchronize()
t0 = time.time()
for i in range(nRound):
    cudart.cudaMemcpy(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    context.execute_v2([int(inputD0), int(outputD0)])
    cudart.cudaMemcpy(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
torch.cuda.synchronize()
time_trt = (time.time() - t0) * 1000 / nRound
print('TRT time: ', time_trt)
# throughput_trt = 1000 / time_trt * batch_size
# print('TRT throughput: ', throughput_trt)

print('****Average diff between trt fp32 and pytorch****')
print(tableHead)
timePerInference = time_trt
check0 = check(outputH0,output_pytorch.detach().cpu().numpy(),True,5e-5)
string = "%4d,%6d,%6d,%8.3f,%9.3e,%9.3e,%9.3e,%9.3e,%9.3e,%9.3e,%9.3e"%(batchSize,
                                                                        width,
                                                                        height,
                                                                        timePerInference,
                                                                        batchSize/timePerInference*1000,
                                                                        check0[1],
                                                                        check0[2],
                                                                        check0[3],
                                                                        check0[4],
                                                                        check0[5],
                                                                        check0[6])
print(string)
# print(string + ", %s"%("Good" if check0[1] < 3.5e-2 and check0[2] < 2e-3 and check1[2] < 1e-1 else "Bad"))

# test TF32 tensorrt performance
print('====', 'TensorRT tf32', '====')
logger = trt.Logger(trt.Logger.ERROR)
with open('elan_x4_tf32.plan', 'rb') as f:
    engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
if engine == None:
    print("Failed loading engine!")
    exit()
print("Succeeded loading engine!")

context = engine.create_execution_context()
context.set_binding_shape(0, [1, 3, 304, 208])
print("EngineBinding0->", engine.get_binding_shape(0), engine.get_binding_dtype(0))
print("EngineBinding1->", engine.get_binding_shape(1), engine.get_binding_dtype(1))

inputH0 = np.ascontiguousarray(test_lr.numpy().reshape(-1))
outputH0 = np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1)))
_, inputD0 = cudart.cudaMalloc(inputH0.nbytes)
_, outputD0 = cudart.cudaMalloc(outputH0.nbytes)

nWarmRound = 10
for i in range(nWarmRound):
    cudart.cudaMemcpy(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    context.execute_v2([int(inputD0), int(outputD0)])
    cudart.cudaMemcpy(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

nRound = 100
torch.cuda.synchronize()
t0 = time.time()
for i in range(nRound):
    cudart.cudaMemcpy(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    context.execute_v2([int(inputD0), int(outputD0)])
    cudart.cudaMemcpy(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
torch.cuda.synchronize()
time_trt = (time.time() - t0) * 1000 / nRound
print('TRT time: ', time_trt)
# throughput_trt = 1000 / time_trt * batch_size
# print('TRT throughput: ', throughput_trt)

print('****Average diff between trt tf32 and pytorch****')
print(tableHead)
timePerInference = time_trt
check0 = check(outputH0,output_pytorch.detach().cpu().numpy(),True,5e-5)
string = "%4d,%6d,%6d,%8.3f,%9.3e,%9.3e,%9.3e,%9.3e,%9.3e,%9.3e,%9.3e"%(batchSize,
                                                                        width,
                                                                        height,
                                                                        timePerInference,
                                                                        batchSize/timePerInference*1000,
                                                                        check0[1],
                                                                        check0[2],
                                                                        check0[3],
                                                                        check0[4],
                                                                        check0[5],
                                                                        check0[6])
print(string)
# print(string + ", %s"%("Good" if check0[1] < 3.5e-2 and check0[2] < 2e-3 and check1[2] < 1e-1 else "Bad"))

cudart.cudaFree(inputD0)
cudart.cudaFree(outputD0)

# test FP16 tensorrt performance
print('====', 'TensorRT fp16', '====')
logger = trt.Logger(trt.Logger.ERROR)
with open('elan_x4_fp16.plan', 'rb') as f:
    engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
if engine == None:
    print("Failed loading engine!")
    exit()
print("Succeeded loading engine!")


context = engine.create_execution_context()
context.set_binding_shape(0, [1, 3, 304, 208])
print("EngineBinding0->", engine.get_binding_shape(0), engine.get_binding_dtype(0))
print("EngineBinding1->", engine.get_binding_shape(1), engine.get_binding_dtype(1))

inputH0 = np.ascontiguousarray(test_lr.numpy().reshape(-1))
outputH0 = np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1)))
_, inputD0 = cudart.cudaMalloc(inputH0.nbytes)
_, outputD0 = cudart.cudaMalloc(outputH0.nbytes)

nWarmRound = 10
for i in range(nWarmRound):
    cudart.cudaMemcpy(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    context.execute_v2([int(inputD0), int(outputD0)])
    cudart.cudaMemcpy(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

nRound = 100
torch.cuda.synchronize()
t0 = time.time()
for i in range(nRound):
    cudart.cudaMemcpy(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    context.execute_v2([int(inputD0), int(outputD0)])
    cudart.cudaMemcpy(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
torch.cuda.synchronize()
time_trt = (time.time() - t0) * 1000 / nRound
print('TRT time: ', time_trt)
# throughput_trt = 1000 / time_trt * batch_size
# print('TRT throughput: ', throughput_trt)

print('****Average diff between trt fp16 and pytorch****')
print(tableHead)
timePerInference = time_trt
check0 = check(outputH0,output_pytorch.detach().cpu().numpy(),True,5e-5)
string = "%4d,%6d,%6d,%8.3f,%9.3e,%9.3e,%9.3e,%9.3e,%9.3e,%9.3e,%9.3e"%(batchSize,
                                                                        width,
                                                                        height,
                                                                        timePerInference,
                                                                        batchSize/timePerInference*1000,
                                                                        check0[1],
                                                                        check0[2],
                                                                        check0[3],
                                                                        check0[4],
                                                                        check0[5],
                                                                        check0[6])
print(string)
# print(string + ", %s"%("Good" if check0[1] < 3.5e-2 and check0[2] < 2e-3 and check1[2] < 1e-1 else "Bad"))


cudart.cudaFree(inputD0)
cudart.cudaFree(outputD0)



# test FP16 tensorrt performance
print('====', 'TensorRT 全锁 fp16', '====')
logger = trt.Logger(trt.Logger.ERROR)
with open('plans/elan_x4_to_fp16.plan', 'rb') as f:
    engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
if engine == None:
    print("Failed loading engine!")
    exit()
print("Succeeded loading engine!")
context = engine.create_execution_context()
context.set_binding_shape(0, [1, 3, 304, 208])
print("EngineBinding0->", engine.get_binding_shape(0), engine.get_binding_dtype(0))
print("EngineBinding1->", engine.get_binding_shape(1), engine.get_binding_dtype(1))

inputH0 = np.ascontiguousarray(test_lr.numpy().reshape(-1))
outputH0 = np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1)))
_, inputD0 = cudart.cudaMalloc(inputH0.nbytes)
_, outputD0 = cudart.cudaMalloc(outputH0.nbytes)

nWarmRound = 10
for i in range(nWarmRound):
    cudart.cudaMemcpy(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    context.execute_v2([int(inputD0), int(outputD0)])
    cudart.cudaMemcpy(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

nRound = 100
torch.cuda.synchronize()
t0 = time.time()
for i in range(nRound):
    cudart.cudaMemcpy(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    context.execute_v2([int(inputD0), int(outputD0)])
    cudart.cudaMemcpy(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
torch.cuda.synchronize()
time_trt = (time.time() - t0) * 1000 / nRound
print('TRT time: ', time_trt)
# throughput_trt = 1000 / time_trt * batch_size
# print('TRT throughput: ', throughput_trt)

print('****Average diff between trt fp16 and pytorch****')
print(tableHead)
timePerInference = time_trt
check0 = check(outputH0,output_pytorch.detach().cpu().numpy(),True,5e-5)
string = "%4d,%6d,%6d,%8.3f,%9.3e,%9.3e,%9.3e,%9.3e,%9.3e,%9.3e,%9.3e"%(batchSize,
                                                                        width,
                                                                        height,
                                                                        timePerInference,
                                                                        batchSize/timePerInference*1000,
                                                                        check0[1],
                                                                        check0[2],
                                                                        check0[3],
                                                                        check0[4],
                                                                        check0[5],
                                                                        check0[6])
print(string)
# print(string + ", %s"%("Good" if check0[1] < 3.5e-2 and check0[2] < 2e-3 and check1[2] < 1e-1 else "Bad"))


cudart.cudaFree(inputD0)
cudart.cudaFree(outputD0)




# test FP16 tensorrt performance
print('====', 'TensorRT 全锁 fp32', '====')
logger = trt.Logger(trt.Logger.ERROR)
with open('plans/elan_x4_to_fp32.plan', 'rb') as f:
    engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
if engine == None:
    print("Failed loading engine!")
    exit()
print("Succeeded loading engine!")
context = engine.create_execution_context()
context.set_binding_shape(0, [1, 3, 304, 208])
print("EngineBinding0->", engine.get_binding_shape(0), engine.get_binding_dtype(0))
print("EngineBinding1->", engine.get_binding_shape(1), engine.get_binding_dtype(1))

inputH0 = np.ascontiguousarray(test_lr.numpy().reshape(-1))
outputH0 = np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1)))
_, inputD0 = cudart.cudaMalloc(inputH0.nbytes)
_, outputD0 = cudart.cudaMalloc(outputH0.nbytes)

nWarmRound = 10
for i in range(nWarmRound):
    cudart.cudaMemcpy(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    context.execute_v2([int(inputD0), int(outputD0)])
    cudart.cudaMemcpy(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

nRound = 100
torch.cuda.synchronize()
t0 = time.time()
for i in range(nRound):
    cudart.cudaMemcpy(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    context.execute_v2([int(inputD0), int(outputD0)])
    cudart.cudaMemcpy(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
torch.cuda.synchronize()
time_trt = (time.time() - t0) * 1000 / nRound
print('TRT time: ', time_trt)
# throughput_trt = 1000 / time_trt * batch_size
# print('TRT throughput: ', throughput_trt)

print('****Average diff between trt fp16 and pytorch****')
print(tableHead)
timePerInference = time_trt
check0 = check(outputH0,output_pytorch.detach().cpu().numpy(),True,5e-5)
string = "%4d,%6d,%6d,%8.3f,%9.3e,%9.3e,%9.3e,%9.3e,%9.3e,%9.3e,%9.3e"%(batchSize,
                                                                        width,
                                                                        height,
                                                                        timePerInference,
                                                                        batchSize/timePerInference*1000,
                                                                        check0[1],
                                                                        check0[2],
                                                                        check0[3],
                                                                        check0[4],
                                                                        check0[5],
                                                                        check0[6])
print(string)
# print(string + ", %s"%("Good" if check0[1] < 3.5e-2 and check0[2] < 2e-3 and check1[2] < 1e-1 else "Bad"))


cudart.cudaFree(inputD0)
cudart.cudaFree(outputD0)







# test FP16 tensorrt performance
print('====', 'TensorRT 卷积锁 fp16', '====')
logger = trt.Logger(trt.Logger.ERROR)
with open('plans/elan_x4_to_fp16_conv.plan', 'rb') as f:
    engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
if engine == None:
    print("Failed loading engine!")
    exit()
print("Succeeded loading engine!")
context = engine.create_execution_context()
context.set_binding_shape(0, [1, 3, 304, 208])
print("EngineBinding0->", engine.get_binding_shape(0), engine.get_binding_dtype(0))
print("EngineBinding1->", engine.get_binding_shape(1), engine.get_binding_dtype(1))

inputH0 = np.ascontiguousarray(test_lr.numpy().reshape(-1))
outputH0 = np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1)))
_, inputD0 = cudart.cudaMalloc(inputH0.nbytes)
_, outputD0 = cudart.cudaMalloc(outputH0.nbytes)

nWarmRound = 10
for i in range(nWarmRound):
    cudart.cudaMemcpy(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    context.execute_v2([int(inputD0), int(outputD0)])
    cudart.cudaMemcpy(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

nRound = 100
torch.cuda.synchronize()
t0 = time.time()
for i in range(nRound):
    cudart.cudaMemcpy(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    context.execute_v2([int(inputD0), int(outputD0)])
    cudart.cudaMemcpy(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
torch.cuda.synchronize()
time_trt = (time.time() - t0) * 1000 / nRound
print('TRT time: ', time_trt)
# throughput_trt = 1000 / time_trt * batch_size
# print('TRT throughput: ', throughput_trt)

print('****Average diff between trt fp16 and pytorch****')
print(tableHead)
timePerInference = time_trt
check0 = check(outputH0,output_pytorch.detach().cpu().numpy(),True,5e-5)
string = "%4d,%6d,%6d,%8.3f,%9.3e,%9.3e,%9.3e,%9.3e,%9.3e,%9.3e,%9.3e"%(batchSize,
                                                                        width,
                                                                        height,
                                                                        timePerInference,
                                                                        batchSize/timePerInference*1000,
                                                                        check0[1],
                                                                        check0[2],
                                                                        check0[3],
                                                                        check0[4],
                                                                        check0[5],
                                                                        check0[6])
print(string)
# print(string + ", %s"%("Good" if check0[1] < 3.5e-2 and check0[2] < 2e-3 and check1[2] < 1e-1 else "Bad"))


cudart.cudaFree(inputD0)
cudart.cudaFree(outputD0)







# test FP16 tensorrt performance
print('====', 'TensorRT 卷积锁 fp32', '====')
logger = trt.Logger(trt.Logger.ERROR)
with open('plans/elan_x4_to_fp32_conv.plan', 'rb') as f:
    engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
if engine == None:
    print("Failed loading engine!")
    exit()
print("Succeeded loading engine!")
context = engine.create_execution_context()
context.set_binding_shape(0, [1, 3, 304, 208])
print("EngineBinding0->", engine.get_binding_shape(0), engine.get_binding_dtype(0))
print("EngineBinding1->", engine.get_binding_shape(1), engine.get_binding_dtype(1))

inputH0 = np.ascontiguousarray(test_lr.numpy().reshape(-1))
outputH0 = np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1)))
_, inputD0 = cudart.cudaMalloc(inputH0.nbytes)
_, outputD0 = cudart.cudaMalloc(outputH0.nbytes)

nWarmRound = 10
for i in range(nWarmRound):
    cudart.cudaMemcpy(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    context.execute_v2([int(inputD0), int(outputD0)])
    cudart.cudaMemcpy(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

nRound = 100
torch.cuda.synchronize()
t0 = time.time()
for i in range(nRound):
    cudart.cudaMemcpy(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    context.execute_v2([int(inputD0), int(outputD0)])
    cudart.cudaMemcpy(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
torch.cuda.synchronize()
time_trt = (time.time() - t0) * 1000 / nRound
print('TRT time: ', time_trt)
# throughput_trt = 1000 / time_trt * batch_size
# print('TRT throughput: ', throughput_trt)

print('****Average diff between trt fp16 and pytorch****')
print(tableHead)
timePerInference = time_trt
check0 = check(outputH0,output_pytorch.detach().cpu().numpy(),True,5e-5)
string = "%4d,%6d,%6d,%8.3f,%9.3e,%9.3e,%9.3e,%9.3e,%9.3e,%9.3e,%9.3e"%(batchSize,
                                                                        width,
                                                                        height,
                                                                        timePerInference,
                                                                        batchSize/timePerInference*1000,
                                                                        check0[1],
                                                                        check0[2],
                                                                        check0[3],
                                                                        check0[4],
                                                                        check0[5],
                                                                        check0[6])
print(string)
# print(string + ", %s"%("Good" if check0[1] < 3.5e-2 and check0[2] < 2e-3 and check1[2] < 1e-1 else "Bad"))


cudart.cudaFree(inputD0)
cudart.cudaFree(outputD0)







# test FP16 tensorrt performance
print('====', 'TensorRT 矩阵乘锁 fp16', '====')
logger = trt.Logger(trt.Logger.ERROR)
with open('plans/elan_x4_to_fp16_matm.plan', 'rb') as f:
    engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
if engine == None:
    print("Failed loading engine!")
    exit()
print("Succeeded loading engine!")
context = engine.create_execution_context()
context.set_binding_shape(0, [1, 3, 304, 208])
print("EngineBinding0->", engine.get_binding_shape(0), engine.get_binding_dtype(0))
print("EngineBinding1->", engine.get_binding_shape(1), engine.get_binding_dtype(1))

inputH0 = np.ascontiguousarray(test_lr.numpy().reshape(-1))
outputH0 = np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1)))
_, inputD0 = cudart.cudaMalloc(inputH0.nbytes)
_, outputD0 = cudart.cudaMalloc(outputH0.nbytes)

nWarmRound = 10
for i in range(nWarmRound):
    cudart.cudaMemcpy(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    context.execute_v2([int(inputD0), int(outputD0)])
    cudart.cudaMemcpy(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

nRound = 100
torch.cuda.synchronize()
t0 = time.time()
for i in range(nRound):
    cudart.cudaMemcpy(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    context.execute_v2([int(inputD0), int(outputD0)])
    cudart.cudaMemcpy(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
torch.cuda.synchronize()
time_trt = (time.time() - t0) * 1000 / nRound
print('TRT time: ', time_trt)
# throughput_trt = 1000 / time_trt * batch_size
# print('TRT throughput: ', throughput_trt)

print('****Average diff between trt fp16 and pytorch****')
print(tableHead)
timePerInference = time_trt
check0 = check(outputH0,output_pytorch.detach().cpu().numpy(),True,5e-5)
string = "%4d,%6d,%6d,%8.3f,%9.3e,%9.3e,%9.3e,%9.3e,%9.3e,%9.3e,%9.3e"%(batchSize,
                                                                        width,
                                                                        height,
                                                                        timePerInference,
                                                                        batchSize/timePerInference*1000,
                                                                        check0[1],
                                                                        check0[2],
                                                                        check0[3],
                                                                        check0[4],
                                                                        check0[5],
                                                                        check0[6])
print(string)
# print(string + ", %s"%("Good" if check0[1] < 3.5e-2 and check0[2] < 2e-3 and check1[2] < 1e-1 else "Bad"))


cudart.cudaFree(inputD0)
cudart.cudaFree(outputD0)







# test FP16 tensorrt performance
print('====', 'TensorRT 矩阵乘锁 fp32', '====')
logger = trt.Logger(trt.Logger.ERROR)
with open('plans/elan_x4_to_fp32_matm.plan', 'rb') as f:
    engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
if engine == None:
    print("Failed loading engine!")
    exit()
print("Succeeded loading engine!")
context = engine.create_execution_context()
context.set_binding_shape(0, [1, 3, 304, 208])
print("EngineBinding0->", engine.get_binding_shape(0), engine.get_binding_dtype(0))
print("EngineBinding1->", engine.get_binding_shape(1), engine.get_binding_dtype(1))

inputH0 = np.ascontiguousarray(test_lr.numpy().reshape(-1))
outputH0 = np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1)))
_, inputD0 = cudart.cudaMalloc(inputH0.nbytes)
_, outputD0 = cudart.cudaMalloc(outputH0.nbytes)

nWarmRound = 10
for i in range(nWarmRound):
    cudart.cudaMemcpy(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    context.execute_v2([int(inputD0), int(outputD0)])
    cudart.cudaMemcpy(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

nRound = 100
torch.cuda.synchronize()
t0 = time.time()
for i in range(nRound):
    cudart.cudaMemcpy(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    context.execute_v2([int(inputD0), int(outputD0)])
    cudart.cudaMemcpy(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
torch.cuda.synchronize()
time_trt = (time.time() - t0) * 1000 / nRound
print('TRT time: ', time_trt)
# throughput_trt = 1000 / time_trt * batch_size
# print('TRT throughput: ', throughput_trt)

print('****Average diff between trt fp16 and pytorch****')
print(tableHead)
timePerInference = time_trt
check0 = check(outputH0,output_pytorch.detach().cpu().numpy(),True,5e-5)
string = "%4d,%6d,%6d,%8.3f,%9.3e,%9.3e,%9.3e,%9.3e,%9.3e,%9.3e,%9.3e"%(batchSize,
                                                                        width,
                                                                        height,
                                                                        timePerInference,
                                                                        batchSize/timePerInference*1000,
                                                                        check0[1],
                                                                        check0[2],
                                                                        check0[3],
                                                                        check0[4],
                                                                        check0[5],
                                                                        check0[6])
print(string)
# print(string + ", %s"%("Good" if check0[1] < 3.5e-2 and check0[2] < 2e-3 and check1[2] < 1e-1 else "Bad"))


cudart.cudaFree(inputD0)
cudart.cudaFree(outputD0)


