import argparse
import yaml
from collections import OrderedDict

import torch
import torch.nn as nn

import utils


parser = argparse.ArgumentParser(description='ELAN')
parser.add_argument('--config', type=str, default=None, help='config file for evaluation and training')

if __name__ == "__main__":
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

    model = utils.import_module('models.{}_network'.format(args.model, args.model)).create_model(args)
    model = model.to(device)
    # Loading weights
    print('load pretrained model: {}!'.format(args.pretrain))
    ckpt = torch.load(args.pretrain)
    new_state_dict = OrderedDict()
    for k, v in ckpt['model_state_dict'].items():
        name = k[7:]
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    print(model)
    model.eval()
    
    # export
    input_data = torch.randn(1, 3, 64, 64, dtype=torch.float32, device='cuda')
    
    input_names = ['lr']
    output_names = ['hr']
    
    torch.onnx.export(model,
                      input_data,
                      'elan_x4_fixed.onnx',
                      input_names=input_names,
                      output_names=output_names,
                      verbose=True,
                      opset_version=15)