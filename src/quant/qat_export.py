import argparse
import yaml
from collections import OrderedDict

import torch
import torch.nn as nn

import pytorch_quantization as qnn

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

    model = utils.import_module('models.qdqelan_network'.format(args.model, args.model)).create_model(args)
    # Loading weights
    print('load pretrained model: {}!'.format(args.pretrain))
    ckpt = torch.load(args.pretrain)
    
    model.load_state_dict(ckpt['model_state_dict'])
    print(model)
    model.eval()
    
    # export
    input_data = torch.randn(1, 3, 304, 208, dtype=torch.float32, device='cuda')
    
    input_names = ['lr']
    output_names = ['hr']
    
    qnn.TensorQuantizer.use_fb_fake_quant = True
    torch.onnx.export(model,
                      input_data,
                      'elan_x4_int8_qat.onnx',
                      input_names=input_names,
                      output_names=output_names,
                      verbose=True,
                      opset_version=15,
                      dynamic_axes={"lr": {0: "batch_size", 2: "width", 3: "height"},
                                    "hr": {0: "batch_size", 2: "width", 3: "height"}})
    print("Succeeded converting model into onnx!")