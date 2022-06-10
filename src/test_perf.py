import argparse
import yaml
import time

import torch
import torch.nn as nn
import numpy as np

import onnx
import onnxruntime as ort

import utils


parser = argparse.ArgumentParser(description='ELAN')
parser.add_argument('--config', type=str, default=None, help='config file for evaluation and training')

# prepare test data
batch_size = 1
channel = 3
height = 80
width = 80

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

model = utils.import_module('models.{}_network'.format(args.model, args.model)).create_model(args)
model = nn.DataParallel(model).to(device)
# Loading weights
print('load pretrained model: {}!'.format(args.pretrain))
ckpt = torch.load(args.pretrain)
model.load_state_dict(ckpt['model_state_dict'])
model = model.eval()

test_lr_cu = test_lr.cuda()
nWarmRound = 10
for i in range(nWarmRound):
    output_pytorch = model(test_lr_cu)

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

# start = torch.cuda.Event(enable_timing=True)
# end = torch.cuda.Event(enable_timing=True)
# stream = torch.cuda.current_stream()
# stream.record_event(start)
# for i in range(nRound):
#     output_pytorch = model(test_lr)
# stream.record_event(end)
# end.synchronize()
# time_pytorch = start.elapsed_time(end) / nRound
# print('Pytorch time: ', time_pytorch)
# throughout_pytorch = 1000 / time_pytorch * batch_size
# print('Pytorch throughout: ', throughout_pytorch)

# time_pytorch_list = []
# for i in range(nRound):
#     start = torch.cuda.Event(enable_timing=True)
#     end = torch.cuda.Event(enable_timing=True)
#     stream = torch.cuda.current_stream()
#     stream.record_event(start)
#     output_pytorch = model(test_lr)
#     stream.record_event(end)
#     end.synchronize()
#     time_pytorch_list.append(start.elapsed_time(end))
# mean_time_pytorch = np.sum(time_pytorch_list) / nRound
# std_time_pytorch = np.std(time_pytorch_list)
# print('Pytorch time: ', mean_time_pytorch)
# print('StdVar: ', std_time_pytorch)
# throughout_pytorch = 1000 / mean_time_pytorch * batch_size
# print('Pytorch throughout: ', throughout_pytorch)

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
print('ONNX time: ', time_onnx)
throughout_onnx = 1000 / time_onnx * batch_size
print('ONNX throughout: ', throughout_onnx)

print('Average diff between onnx and pytorch: ', np.mean(np.abs(output_pytorch.detach().cpu().numpy() - output_onnx[0]) / np.abs(output_pytorch.detach().cpu().numpy())))

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
throughout_onnx_sed = 1000 / time_onnx_sed * batch_size
print('ONNX surgeoned throughout: ', throughout_onnx_sed)

print('Average diff between onnx and onnx_sed: ', np.mean(np.abs(output_onnx[0] - output_onnx_sed[0]) / np.abs(output_onnx[0])))

# test FP32 tensorrt performance

