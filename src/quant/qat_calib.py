import math
import argparse, yaml
import utils
import os
from tqdm import tqdm
import logging
import sys
import time
import importlib
import glob

from torch.autograd import Variable as V

import pytorch_quantization.nn as qnn
import pytorch_quantization.calib as calib
from pytorch_quantization.tensor_quant import QuantDescriptor

calibrator = ["max", "histogram"][1]
nCalibrationBatchSize = 4
percentileList = [99.9, 99.99, 99.999, 99.9999]

parser = argparse.ArgumentParser(description='ELAN')
## yaml configuration files
parser.add_argument('--config', type=str, default=None, help = 'pre-config file for training')
parser.add_argument('--resume', type=str, default=None, help = 'resume training or not')

quant_desc_input = QuantDescriptor(calib_method=calibrator, axis=None)
qnn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
qnn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
quant_desc_weight = QuantDescriptor(calib_method=calibrator, axis=None)
qnn.QuantConv2d.set_default_quant_desc_weight(quant_desc_weight)
qnn.QuantLinear.set_default_quant_desc_weight(quant_desc_weight)

if __name__ == '__main__':
    args = parser.parse_args()
    if args.config:
       opt = vars(args)
       yaml_args = yaml.load(open(args.config), Loader=yaml.FullLoader)
       opt.update(yaml_args)
    ## set visibel gpu   
    gpu_ids_str = str(args.gpu_ids).replace('[','').replace(']','')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_ids_str)
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim.lr_scheduler import MultiStepLR, StepLR
    from datas.utils import create_datasets


    ## select active gpu devices
    device = None
    if args.gpu_ids is not None and torch.cuda.is_available():
        print('use cuda & cudnn for acceleration!')
        print('the gpu id is: {}'.format(args.gpu_ids))
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        print('use cpu for training!')
        device = torch.device('cpu')
    torch.set_num_threads(args.threads)

    ## create dataset for training and validating
    train_dataloader, valid_dataloaders = create_datasets(args)

    ## definitions of model
    try:
        model_raw = utils.import_module('models.qdqelan_network'.format(args.model, args.model)).create_model(args)
    except Exception:
        raise ValueError('not supported model type! or something')
    model = nn.DataParallel(model_raw).to(device)

    ## definition of loss and optimizer
    loss_func = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=args.decays, gamma=args.gamma)

    ## load pretrain
    if args.pretrain is not None:
        print('load pretrained model: {}!'.format(args.pretrain))
        ckpt = torch.load(args.pretrain)
        model.load_state_dict(ckpt['model_state_dict'])
        
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, qnn.TensorQuantizer):
                if module._calibrator is not None:
                    module.disable_quant()
                    module.enable_calib()
                else:
                    module.disable()
                    
        for i, batch in enumerate(train_dataloader):
            
            lr, hr = batch
            H, W = lr.shape[2:]
            lr = model_raw.check_image_size(lr)
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)
            sr = sr[:, :, 0:H*model_raw.scale, 0:W*model_raw.scale]
            if i >= nCalibrationBatchSize:
                break
        
        for name, module in model.named_modules():
            if isinstance(module, qnn.TensorQuantizer):
                if module._calibrator is not None:
                    module.enable_quant()
                    module.disable_calib()
                else:
                    module.enable()
                    
        def computeArgMax(model, **kwargs):
            for name, module in model.named_modules():
                if isinstance(module, qnn.TensorQuantizer) and module._calibrator is not None:
                    if isinstance(module._calibrator, calib.MaxCalibrator):
                        module.load_calib_amax()
                    else:
                        module.load_calib_amax(**kwargs)
                    
        if calibrator == "max":
            computeArgMax(model, method="max")
            modelName = "./model-max-%d.pth"%(nCalibrationBatchSize * train_dataloader.batch_size)
            
        else:
            for percentile in percentileList:
                computeArgMax(model, method="percentile")
                modelName = "./model-percentile-%f-%d.pth"%(percentile,nCalibrationBatchSize * train_dataloader.batch_size)

            for method in ["mse", "entropy"]:
                computeArgMax(model, method=method)
                modelName = "./model-%s-%f.pth"%(method, percentile)
                
    print("Succeeded calibrating model in pyTorch!")
    
    # finetune in pyTorch
    model.cuda()
    
    for epoch in range(1):
        epoch_loss = 0.0
        model = model.train()
        opt_lr = scheduler.get_last_lr()
        print('##==========={}-training, Epoch: {}, lr: {} =============##'.format('fp32', epoch, opt_lr))
        for iter, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            lr, hr = batch
            lr = model_raw.check_image_size(lr)
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)
            loss = loss_func(sr, hr)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss)

            if (iter + 1) % args.log_every == 0:
                timer_start = time.time()
                cur_steps = (iter+1)*args.batch_size
                total_steps = len(train_dataloader.dataset)
                fill_width = math.ceil(math.log10(total_steps))
                cur_steps = str(cur_steps).zfill(fill_width)

                epoch_width = math.ceil(math.log10(args.epochs))
                cur_epoch = str(epoch).zfill(epoch_width)

                avg_loss = epoch_loss / (iter + 1)

                timer_end = time.time()
                duration = timer_end - timer_start
                timer_start = timer_end
                print('Epoch:{}, {}/{}, loss: {:.4f}, time: {:.3f}'.format(cur_epoch, cur_steps, total_steps, avg_loss, duration))

        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
                }, modelName)
    
    # eval
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

    # export to ONNX
    model.eval()
    qnn.TensorQuantizer.use_fb_fake_quant = True
    input_data = torch.randn(1, 3, 80, 80, dtype=torch.float32, device='cuda')
    
    input_names = ['lr']
    output_names = ['hr']
    
    torch.onnx.export(model.module,
                      input_data,
                      'elan_x4_qat_1f1.onnx',
                      input_names=input_names,
                      output_names=output_names,
                      do_constant_folding=True,
                      verbose=True,
                      keep_initializers_as_inputs=True,
                      opset_version=15,
                      dynamic_axes={"lr": {0: "batch_size", 2: "width", 3: "height"},
                                    "hr": {0: "batch_size", 2: "width", 3: "height"}})
    
    print("Succeeded converting model into onnx!")