import os
import sys
from glob import glob
from datetime import datetime as dt

import torch
import torch.nn.functional as F
from torch.autograd import Variable as V
import numpy as np
import cv2
import pytorch_quantization.nn as qnn
import pytorch_quantization.calib as calib
from pytorch_quantization.tensor_quant import QuantDescriptor


dataPath = os.path.dirname(os.path.realpath(__file__)) + "/../00-MNISTData/"
sys.path.append(dataPath)
import loadMnistData

torch.manual_seed(97)
np.random.seed(97)

nImageHeight = 28
nImageWidth = 28
nTrainBatchSize = 128
nCalibrationBatchSize = 4
calibrator = ["max", "histogram"][1]
percentileList = [99.9, 99.99, 99.999, 99.9999]
onnxFile = 'model.onnx'
trtFile = 'model.plan'

quant_desc_input = QuantDescriptor(calib_method=calibrator, axis=None)
qnn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
qnn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
quant_desc_weight = QuantDescriptor(calib_method=calibrator, axis=None)
qnn.QuantConv2d.set_default_quant_desc_weight(quant_desc_weight)
qnn.QuantLinear.set_default_quant_desc_weight(quant_desc_weight)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = qnn.QuantConv2d(1, 32, (5, 5), padding=(2, 2), bias=True)
        self.conv2 = qnn.QuantConv2d(32, 64, (5, 5), padding=(2, 2), bias=True)
        self.fc1 = qnn.QuantLinear(64 * 7 * 7, 1024, bias=True)
        self.fc2 = qnn.QuantLinear(1024, 10, bias=True)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.reshape(-1, 7 * 7 * 64)
        x = F.relu(self.fc1(x))
        y = self.fc2(x)
        z = F.softmax(y, dim=1)
        z = torch.argmax(z, dim=1)
        return y, z
    

class MyData(torch.utils.data.Dataset):

    def __init__(self, path=dataPath, isTrain=True, nTrain=0, nTest=0):
        if isTrain:
            if len(glob(dataPath + "train/*.jpg")) == 0:
                mnist = loadMnistData.MnistData(path, isOneHot=False)
                mnist.saveImage([60000, nTrain][int(nTrain > 0)], path + "train/", True)
            self.data = glob(path + "train/*.jpg")
        else:
            if len(glob(dataPath + "test/*.jpg")) == 0:
                mnist = loadMnistData.MnistData(path, isOneHot=False)
                mnist.saveImage([10000, nTest][int(nTest > 0)], path + "test/", False)
            self.data = glob(path + "test/*.jpg")

    def __getitem__(self, index):
        imageName = self.data[index]
        data = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE)
        label = np.zeros(10, dtype=np.float32)
        index = int(imageName[-7])
        label[index] = 1
        return torch.from_numpy(data.reshape(1, nImageHeight, nImageWidth).astype(np.float32)), label

    def __len__(self):
        return len(self.data)
    
# create network with PyTorch
model = Net().cuda()
ceLoss = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.001)
trainDataset = MyData(isTrain=True, nTrain=600)
testDataset = MyData(isTrain=False, nTest=100)
trainLoader = torch.utils.data.DataLoader(dataset=trainDataset, batch_size=nTrainBatchSize, shuffle=True)
testLoader = torch.utils.data.DataLoader(dataset=testDataset, batch_size=nTrainBatchSize, shuffle=True)

for epoch in range(40):
    for i, (xTrain, yTrain) in enumerate(trainLoader):
        xTrain = V(xTrain).cuda()
        yTrain = V(yTrain).cuda()
        opt.zero_grad()
        y_, z = model(xTrain)
        loss = ceLoss(y_, yTrain)
        loss.backward()
        opt.step()
    if not (epoch + 1) % 10:
        print("%s, epoch %d, loss = %f" % (dt.now(), epoch + 1, loss.data))
        

acc = 0
model.eval()
for xTest, yTest in testLoader:
    xTest = V(xTest).cuda()
    yTest = V(yTest).cuda()
    y_, z = model(xTest)
    acc += torch.sum(z == torch.matmul(yTest, torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).to('cuda:0'))).cpu().numpy()
print("test acc = %f" % (acc / len(testLoader) / nTrainBatchSize))
print("Succeeded building model in pyTorch!")

# aclibrate in PyTroch
with torch.no_grad():
    for name, module in model.named_modules():
        if isinstance(module, qnn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()
                
    for i, (xTrain, yTrain) in enumerate(trainLoader):
        model(V(xTrain).cuda())
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
        modelName = "./model-max-%d.pth"%(nCalibrationBatchSize * trainLoader.batch_size)
        
    else:
        for percentile in percentileList:
            computeArgMax(model, method="percentile")
            modelName = "./model-percentile-%f-%d.pth"%(percentile,nCalibrationBatchSize * trainLoader.batch_size)

        for method in ["mse", "entropy"]:
            computeArgMax(model, method=method)
            modelName = "./model-%s-%f.pth"%(method, percentile)
            
print("Succeeded calibrating model in pyTorch!")

# finetune in pyTorch
model.cuda()

for epoch in range(10):
    for i, (xTrain, yTrain) in enumerate(trainLoader):
        xTrain = V(xTrain).cuda()
        yTrain = V(yTrain).cuda()
        opt.zero_grad()
        y_, z = model(xTrain)
        loss = ceLoss(y_, yTrain)
        loss.backward()
        opt.step()
    print("%s, epoch %d, loss = %f" % (dt.now(), epoch, loss.data))
    
acc = 0
model.eval()
for xTest, yTest in testLoader:
    xTest = V(xTest).cuda()
    yTest = V(yTest).cuda()
    y_, z = model(xTest)
    acc += torch.sum(z == torch.matmul(yTest, torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).to('cuda:0'))).cpu().numpy()
print("test acc = %f" % (acc / len(testLoader) / nTrainBatchSize))
print("Succeeded fine tuning model in pyTorch!")

# export to ONNX
model.eval()
qnn.TensorQuantizer.use_fb_fake_quant = True
torch.onnx.export(model,
                  torch.randn(1, 1, nImageHeight, nImageWidth, device="cuda"),
                  onnxFile,
                  input_names=['x'],
                  output_names=['y', 'z'],
                  do_constant_folding=True,
                  verbose=True,
                  keep_initializers_as_inputs=True,
                  opset_version=13,
                  dynamic_axes={"x": {0: "nBatchSize"}})
print("Succeeded converting model into onnx!")

# 