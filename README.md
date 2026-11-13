# Stanford CS231n: Convolutional Neural Networks for Visual Recognition

![Stanford](https://img.shields.io/badge/Course-Stanford%20CS231n-red)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c)
![NumPy](https://img.shields.io/badge/From%20Scratch-NumPy-013243)
![Status](https://img.shields.io/badge/Status-All%20Assignments%20Completed-success)

本仓库收录了斯坦福大学经典公开课 **CS231n (Convolutional Neural Networks for Visual Recognition)** 的全部核心作业实现。

整个项目涵盖了从最基础的图像分类到复杂的视觉任务。**核心亮点在于：** 前期的底层算子（如全连接层、卷积层、池化层、BatchNorm、Dropout 等）及优化器（SGD, Adam 等）均**未使用**任何深度学习框架，完全基于 `NumPy` 从零推导矩阵并手写前向与反向传播；而在后期，熟练过渡到了 `PyTorch` 框架，完成了高级视觉任务的工程部署。

## 📁 项目结构

```text
cs231n-assignments/
│
├── assignment1/            # 基础与全连接网络
│   ├── knn.ipynb           # KNN 向量化距离计算与交叉验证
│   ├── softmax.ipynb       # Softmax 损失与梯度推导
│   ├── two_layer_net.ipynb # 两层全连接网络
│   └── features.ipynb      # 基于 HOG 与 HSV 特征的高级分类器
│
├── assignment2/            # 卷积神经网络与框架入门
│   ├── FullyConnectedNets.ipynb # 模块化全连接网与高级优化器
│   ├── BatchNormalization.ipynb # 从零手写 BatchNorm/LayerNorm
│   ├── Dropout.ipynb       # 从零手写 Dropout 正则化
│   ├── ConvolutionalNetworks.ipynb # 从零手写 Conv/Pooling 前后向
│   └── PyTorch.ipynb       # PyTorch 框架实战与模型调优
│
└── assignment3/            # 高级计算机视觉任务
    ├── RNN_Captioning.ipynb # RNN/LSTM 图像描述生成
    ├── Transformer_Captioning.ipynb # Transformer 图像描述
    ├── NetworkVisualization.ipynb # 显著图、风格迁移、对抗样本
    └── Generative_Adversarial_Networks.ipynb # GAN 图像生成
