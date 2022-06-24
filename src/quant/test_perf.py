import argparse
import yaml
import os

import torch
import torch.nn as nn

import numpy as np
from tqdm import tqdm

import tensorrt as trt
from cuda import cudart

from datas.utils import create_datasets
import utils

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

# Create datasets
_, valid_dataloaders = create_datasets(args)


# pytorch original version test
model_raw = utils.import_module('models.{}_network'.format(args.model, args.model)).create_model(args)
model = nn.DataParallel(model_raw).to(device)
# Loading weights
print('load pretrained model: {}!'.format(args.pretrain))
ckpt = torch.load(args.pretrain)
model.load_state_dict(ckpt['model_state_dict'])

torch.set_grad_enabled(False)
model = model.eval()
test_log = ""

for valid_dataloader in valid_dataloaders:
    avg_psnr, avg_ssim = 0.0, 0.0
    name = valid_dataloader['name']
    loader = valid_dataloader['dataloader']
    for lr, hr in tqdm(loader, ncols=80):
        lr, hr = lr.to(device), hr.to(device)
        H, W = lr.shape[2:]
        lr = model_raw.check_image_size(lr)
        sr = model(lr)
        sr = sr[:, :, 0:H*model_raw.scale, 0:W*model_raw.scale]
        # quantize output to [0, 255]
        hr = hr.clamp(0, 255)
        sr = sr.clamp(0, 255)
        # convert to ycbcr
        if args.colors == 3:
            hr_ycbcr = utils.rgb_to_ycbcr(hr)
            sr_ycbcr = utils.rgb_to_ycbcr(sr)
            hr = hr_ycbcr[:, 0:1, :, :]
            sr = sr_ycbcr[:, 0:1, :, :]
        # crop image for evaluation
        hr = hr[:, :, args.scale:-args.scale, args.scale:-args.scale]
        sr = sr[:, :, args.scale:-args.scale, args.scale:-args.scale]
        # calculate psnr and ssim
        psnr = utils.calc_psnr(sr, hr)
        ssim = utils.calc_ssim(sr, hr)
        avg_psnr += psnr
        avg_ssim += ssim
    avg_psnr = round(avg_psnr/len(loader) + 5e-3, 2)
    avg_ssim = round(avg_ssim/len(loader) + 5e-5, 4)
    
    test_log += '[{}-X{}], PSNR/SSIM: {:.2f}/{:.4f}\n'.format(
        name, args.scale, float(avg_psnr), float(avg_ssim)
    )
print(test_log)

# fp32-trt
logger = trt.Logger(trt.Logger.ERROR)
with open('elan_x4.plan', 'rb') as f:
    engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
if engine == None:
    print("Failed loading engine!")
    exit()
print("Succeeded loading engine!")
test_log = ""


for valid_dataloader in valid_dataloaders:
    avg_psnr, avg_ssim = 0.0, 0.0
    name = valid_dataloader['name']
    loader = valid_dataloader['dataloader']
    for lr, hr in tqdm(loader, ncols=80):
        H, W = lr.shape[2:]
        lr = model_raw.check_image_size(lr)
        context = engine.create_execution_context()
        context.set_binding_shape(0, [1, 3, lr.shape[2], lr.shape[3]])
        print("EngineBinding0->", engine.get_binding_shape(0), engine.get_binding_dtype(0))
        print("EngineBinding1->", engine.get_binding_shape(1), engine.get_binding_dtype(1))
        
        inputH0 = np.ascontiguousarray(lr.numpy().reshape(-1))
        outputH0 = np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1)))
        _, inputD0 = cudart.cudaMalloc(inputH0.nbytes)
        _, outputD0 = cudart.cudaMalloc(outputH0.nbytes)

        cudart.cudaMemcpy(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        context.execute_v2([int(inputD0), int(outputD0)])
        cudart.cudaMemcpy(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

        sr = torch.Tensor(outputH0[:, :, 0:H*model_raw.scale, 0:W*model_raw.scale])
        # quantize output to [0, 255]
        hr = hr.clamp(0, 255)
        sr = sr.clamp(0, 255)
        # convert to ycbcr
        if args.colors == 3:
            hr_ycbcr = utils.rgb_to_ycbcr(hr)
            sr_ycbcr = utils.rgb_to_ycbcr(sr)
            hr = hr_ycbcr[:, 0:1, :, :]
            sr = sr_ycbcr[:, 0:1, :, :]
        # crop image for evaluation
        hr = hr[:, :, args.scale:-args.scale, args.scale:-args.scale]
        sr = sr[:, :, args.scale:-args.scale, args.scale:-args.scale]
        # calculate psnr and ssim
        psnr = utils.calc_psnr(sr, hr)
        ssim = utils.calc_ssim(sr, hr)
        avg_psnr += psnr
        avg_ssim += ssim

    avg_psnr = round(avg_psnr/len(loader) + 5e-3, 2)
    avg_ssim = round(avg_ssim/len(loader) + 5e-5, 4)
    
    test_log += '[{}-X{}], PSNR/SSIM: {:.2f}/{:.4f}\n'.format(
        name, args.scale, float(avg_psnr), float(avg_ssim)
    )
print(test_log)

# tf32-trt
logger = trt.Logger(trt.Logger.ERROR)
with open('elan_x4_tf32.plan', 'rb') as f:
    engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
if engine == None:
    print("Failed loading engine!")
    exit()
print("Succeeded loading engine!")
test_log = ""


for valid_dataloader in valid_dataloaders:
    avg_psnr, avg_ssim = 0.0, 0.0
    name = valid_dataloader['name']
    loader = valid_dataloader['dataloader']
    for lr, hr in tqdm(loader, ncols=80):
        H, W = lr.shape[2:]
        lr = model_raw.check_image_size(lr)
        context = engine.create_execution_context()
        context.set_binding_shape(0, [1, 3, lr.shape[2], lr.shape[3]])
        print("EngineBinding0->", engine.get_binding_shape(0), engine.get_binding_dtype(0))
        print("EngineBinding1->", engine.get_binding_shape(1), engine.get_binding_dtype(1))
        
        inputH0 = np.ascontiguousarray(lr.numpy().reshape(-1))
        outputH0 = np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1)))
        _, inputD0 = cudart.cudaMalloc(inputH0.nbytes)
        _, outputD0 = cudart.cudaMalloc(outputH0.nbytes)

        cudart.cudaMemcpy(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        context.execute_v2([int(inputD0), int(outputD0)])
        cudart.cudaMemcpy(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

        sr = torch.Tensor(outputH0[:, :, 0:H*model_raw.scale, 0:W*model_raw.scale])
        # quantize output to [0, 255]
        hr = hr.clamp(0, 255)
        sr = sr.clamp(0, 255)
        # convert to ycbcr
        if args.colors == 3:
            hr_ycbcr = utils.rgb_to_ycbcr(hr)
            sr_ycbcr = utils.rgb_to_ycbcr(sr)
            hr = hr_ycbcr[:, 0:1, :, :]
            sr = sr_ycbcr[:, 0:1, :, :]
        # crop image for evaluation
        hr = hr[:, :, args.scale:-args.scale, args.scale:-args.scale]
        sr = sr[:, :, args.scale:-args.scale, args.scale:-args.scale]
        # calculate psnr and ssim
        psnr = utils.calc_psnr(sr, hr)
        ssim = utils.calc_ssim(sr, hr)
        avg_psnr += psnr
        avg_ssim += ssim

    avg_psnr = round(avg_psnr/len(loader) + 5e-3, 2)
    avg_ssim = round(avg_ssim/len(loader) + 5e-5, 4)
    
    test_log += '[{}-X{}], PSNR/SSIM: {:.2f}/{:.4f}\n'.format(
        name, args.scale, float(avg_psnr), float(avg_ssim)
    )
print(test_log)


# fp16-trt
logger = trt.Logger(trt.Logger.ERROR)
with open('elan_x4_fp16.plan', 'rb') as f:
    engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
if engine == None:
    print("Failed loading engine!")
    exit()
print("Succeeded loading engine!")
test_log = ""


for valid_dataloader in valid_dataloaders:
    avg_psnr, avg_ssim = 0.0, 0.0
    name = valid_dataloader['name']
    loader = valid_dataloader['dataloader']
    for lr, hr in tqdm(loader, ncols=80):
        H, W = lr.shape[2:]
        lr = model_raw.check_image_size(lr)
        context = engine.create_execution_context()
        context.set_binding_shape(0, [1, 3, lr.shape[2], lr.shape[3]])
        print("EngineBinding0->", engine.get_binding_shape(0), engine.get_binding_dtype(0))
        print("EngineBinding1->", engine.get_binding_shape(1), engine.get_binding_dtype(1))
        
        inputH0 = np.ascontiguousarray(lr.numpy().reshape(-1))
        outputH0 = np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1)))
        _, inputD0 = cudart.cudaMalloc(inputH0.nbytes)
        _, outputD0 = cudart.cudaMalloc(outputH0.nbytes)

        cudart.cudaMemcpy(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        context.execute_v2([int(inputD0), int(outputD0)])
        cudart.cudaMemcpy(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

        sr = torch.Tensor(outputH0[:, :, 0:H*model_raw.scale, 0:W*model_raw.scale])
        # quantize output to [0, 255]
        hr = hr.clamp(0, 255)
        sr = sr.clamp(0, 255)
        # convert to ycbcr
        if args.colors == 3:
            hr_ycbcr = utils.rgb_to_ycbcr(hr)
            sr_ycbcr = utils.rgb_to_ycbcr(sr)
            hr = hr_ycbcr[:, 0:1, :, :]
            sr = sr_ycbcr[:, 0:1, :, :]
        # crop image for evaluation
        hr = hr[:, :, args.scale:-args.scale, args.scale:-args.scale]
        sr = sr[:, :, args.scale:-args.scale, args.scale:-args.scale]
        # calculate psnr and ssim
        psnr = utils.calc_psnr(sr, hr)
        ssim = utils.calc_ssim(sr, hr)
        avg_psnr += psnr
        avg_ssim += ssim

    avg_psnr = round(avg_psnr/len(loader) + 5e-3, 2)
    avg_ssim = round(avg_ssim/len(loader) + 5e-5, 4)
    
    test_log += '[{}-X{}], PSNR/SSIM: {:.2f}/{:.4f}\n'.format(
        name, args.scale, float(avg_psnr), float(avg_ssim)
    )
print(test_log)



# int8-trt
logger = trt.Logger(trt.Logger.ERROR)
with open('elan_x4_int8.plan', 'rb') as f:
    engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
if engine == None:
    print("Failed loading engine!")
    exit()
print("Succeeded loading engine!")
test_log = ""


for valid_dataloader in valid_dataloaders:
    avg_psnr, avg_ssim = 0.0, 0.0
    name = valid_dataloader['name']
    loader = valid_dataloader['dataloader']
    for lr, hr in tqdm(loader, ncols=80):
        H, W = lr.shape[2:]
        lr = model_raw.check_image_size(lr)
        context = engine.create_execution_context()
        context.set_binding_shape(0, [1, 3, lr.shape[2], lr.shape[3]])
        print("EngineBinding0->", engine.get_binding_shape(0), engine.get_binding_dtype(0))
        print("EngineBinding1->", engine.get_binding_shape(1), engine.get_binding_dtype(1))
        
        inputH0 = np.ascontiguousarray(lr.numpy().reshape(-1))
        outputH0 = np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1)))
        _, inputD0 = cudart.cudaMalloc(inputH0.nbytes)
        _, outputD0 = cudart.cudaMalloc(outputH0.nbytes)

        cudart.cudaMemcpy(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        context.execute_v2([int(inputD0), int(outputD0)])
        cudart.cudaMemcpy(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

        sr = torch.Tensor(outputH0[:, :, 0:H*model_raw.scale, 0:W*model_raw.scale])
        # quantize output to [0, 255]
        hr = hr.clamp(0, 255)
        sr = sr.clamp(0, 255)
        # convert to ycbcr
        if args.colors == 3:
            hr_ycbcr = utils.rgb_to_ycbcr(hr)
            sr_ycbcr = utils.rgb_to_ycbcr(sr)
            hr = hr_ycbcr[:, 0:1, :, :]
            sr = sr_ycbcr[:, 0:1, :, :]
        # crop image for evaluation
        hr = hr[:, :, args.scale:-args.scale, args.scale:-args.scale]
        sr = sr[:, :, args.scale:-args.scale, args.scale:-args.scale]
        # calculate psnr and ssim
        psnr = utils.calc_psnr(sr, hr)
        ssim = utils.calc_ssim(sr, hr)
        avg_psnr += psnr
        avg_ssim += ssim

    avg_psnr = round(avg_psnr/len(loader) + 5e-3, 2)
    avg_ssim = round(avg_ssim/len(loader) + 5e-5, 4)
    
    test_log += '[{}-X{}], PSNR/SSIM: {:.2f}/{:.4f}\n'.format(
        name, args.scale, float(avg_psnr), float(avg_ssim)
    )
print(test_log)

