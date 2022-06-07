import argparse
import yaml
import utils
import os

import torch
from tqdm import tqdm

from datas.utils import create_datasets


parser = argparse.ArgumentParser(description='ELAN')
parser.add_argument('--config', type=str, default=None, help='config file for evaluation and training')

if __name__ == "__main__":
    args = parser.parse_args()
    if args.config:
        opt = vars(args)
        yaml_args = yaml.load(open(args.config), loader=yaml.FullLoader)
        opt.update(yaml_args)
    print('use cuda & cudnn for acceleration!')
    print('the gpu id is: {}'.format(args.gpu_ids))
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(args.threads)

    model = utils.import_module('models.{}_network'.format(args.model, args.model)).create_model(args)
    
    # Loading weights
    print('load pretrained model: {}!'.format(args.pretrain))
    ckpt = torch.load(args.pretrain)
    model.load_state_dict(ckpt['model_state_dict'])
    print(model)
    
    # Create datasets
    _, valid_dataloaders = create_datasets(args)
    
    # Evaluating
    torch.set_grad_enabled(False)
    model = model.eval()
    test_log = ""
    
    for valid_dataloader in valid_dataloaders:
        avg_psnr, avg_ssim = 0.0, 0.0
        name = valid_dataloader['name']
        loader = valid_dataloader['dataloader']
        for lr, hr in tqdm(loader, ncols=80):
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)
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