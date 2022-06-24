import time
import argparse
import yaml

import numpy as np
import torch
import torch.nn as nn

import tensorrt as trt
from cuda import cudart

import utils

parser = argparse.ArgumentParser(description='ELAN')
parser.add_argument('--config', type=str, default=None, help='config file for evaluation and training')

batch_size = 1
batchSize = 1
channel = 3
height = 304
width = 208

test_lr = torch.randn([batch_size, channel, height, width], dtype=torch.float32)

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

time.sleep(5)


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


cudart.cudaFree(inputD0)
cudart.cudaFree(outputD0)