# trt-elan
该项目实现了图像超分辨率算法ELAN的TensorRT加速版本。

:construction:**该项目正在施工**。

## 总述

### 原始模型信息

- 模型名称：ELAN，Efficient Long-Range Attention Network for Image Super-resollution，[arxiv link](https://arxiv.org/abs/2203.06697)，[code link](https://github.com/xindongzhang/ELAN)。已投稿ECCV2022。
- 模型任务：图像超分辨率任务
- 模型特点：轻量级模型；在轻量级限制下，击败SwinIR效果；结合使用卷积与自注意力机制打造基于类Transformer结构但更轻量化的超分模型。

### 优化效果

使用超分任务常用验证集Manga109作为测试基准数据集，使用NVIDIA A10作为测试机器。加速比计算使用TensorRT运行时间比上PyTorch运行时间。

- FP32下可以做到无损精度，加速比为1.19
- TF32下可以做到近无损精度，加速比为1.39
- FP16下可以做到验证集无损精度，加速比为2.14
- INT8 QAT下理论可达验证集无损精度，加速比为2.53

### 特性

- [x] 固定尺寸优化
- [ ] 动态尺寸优化
- [x] FP16量化
- [x] 精度优化的FP16量化
- [x] INT8 PTQ
- [x] INT8 QAT
- [ ] LFE Plugin
- [ ] GMSA Plugin

### 项目使用指南

#### 安装



#### 使用



## 原始模型

### 模型简介

#### 模型用途

图片超分辨率，即为低清晰度图片填充细节，使之成为高清晰度图片

#### 模型效果

##### 视觉效果

![](assets/pic.jpeg)

##### 实验效果

此模型的精度超过了目前超分辨率领域顶尖的SwinIR模型，且计算性能极大提升。

![](assets/tab.jpeg)

#### 业界实际运用情况

此模型为目前超分辨率领域模型计算加速方向的最新研究进展，还未在工业界得到应用。
但超分辨率技术已经广泛应用于视频和游戏画质增强中。NVIDIA DLSS(深度学习超采样)便是其中的代表。DLSS利用 GeForce RTX™ GPU 上的专用 AI 处理单元 - Tensor Core 将视觉保真度提升至全新高度。DLSS 利用深度学习神经网络的强大功能提高帧率和分辨率，为游戏生成精美清晰的图像。

#### 模型特色

模型的主体结构于基于Swin Transformer的SwinIR模型相同，但对其中的自注意力计算过程进行了改进：

* 提出一种新的自注意力操作GMSA(Group-wise Multi-scale Self-Attention)
  * 将特征矩阵的各通道划分为不同的组，每个组应用不同的patch大小，最后将计算结果进行拼接，从而可以通过调整各通道对每个组的分配情况灵活地调节计算量，可避免patch过大带来的计算量增长和patch过小导致的输出质量下降
* 改进了现有的自注意力计算过程，使之更适合并行计算
  * 将LayerNorm改为BatchNorm并与卷积操作合并，从而减少element-wise操作
  * 使用同一个操作计算K和Q矩阵，并在对称高斯空间计算相似度，从而减少1x1卷积计算并节约内存
  * 在第一层之后的自注意力层不再计算K和Q矩阵，而用上述第一层K和Q矩阵在对称高斯空间计算得到的相似度结果与各层的V矩阵相乘，从而节约大量计算资源

### 模型优化的难点

* 如上节所述，此模型中所使用的并非标准的Transformer模块，而是针对并行加速进行了一些改进，因此无法使用TensorRT社区中现有的Transformer模块实现。
* 此模型对Transformer模块的改进是专门针对并行加速进行的，因此在TensorRT中有很大的优化空间，虽然trtexec可以完成一部分的优化工作，但由于其计算过程的独特性，许多优化操作仍需要通过编写插件手动进行

## 优化过程

### 预处理部分的不同处理方式

修剪图。

遭遇ReflectPadding不成功支持BUG。

可参考玮神给出的Workaround进行支持。

也可直接跳过对预处理过程的TRT化。

### FP32导出

【描述】

### TF32导出

【描述】


### FP16导出

#### 精度问题分析



#### 精度问题修正



### INT8量化

#### PTQ方案

【描述】

#### QAT方案

【描述】

### Profiling分析

GPU利用率很高



TensorRT起作用的主要因素



开销较大的层



LFE GMSA Plugin化潜力



## 精度与加速效果

使用超分任务常用验证集Manga109作为测试基准数据集，使用NVIDIA A10作为测试机器。加速比计算使用TensorRT运行时间比上PyTorch运行时间。

### 加速效果

【时延柱形图】

【压力测试下折线图】

### 验证集精度

【精度对比表格】

### 张量对齐精度

【张量精度对比表格】

QAT的INT8量化能带来更好的性能，但它需要重新训练模型，这部分工作仍在推进中，故当前本仓库仅提供了QAT的完整代码，但QAT的完整方案还在生成测试中。



## 仍然存在的问题

- 暂时未能支持Dynamic Shape，【描述】

## Bug报告

### ReflectPadding Parse Error

- Environment
  - NVIDIA A10
  - TensorRT 8.4GA
  - CUDA 11.6
  - CuDNN 8.4.0
  - CUBLAS 11.9.2
  - NVIDIA Driver 510.73.08

- Reproduction Steps
  - 运行单元测试程序
- Expected Behavior
  - 根据【官方更新信息】，应该可以正确导出该OP，且支持reflect模式
- Actual Behavior
  - Error
- Additional Notes
  - 可暂时参考玮神给出的workaround解决该问题。可以对应支持到一些未能给出解答的帖子。

## 经验与体会






## 相关的项目

- ELAN算法的官方仓库：[xindongzhang](https://github.com/xindongzhang)/[ELAN](https://github.com/xindongzhang/ELAN)

