# -*- coding:utf-8 -*-
# @Time : 2021/3/28 14:56
# @Author: LCHJ
# @File : resnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=1, bias=False)


class residual_block(nn.Module):
    
    def __init__(self, in_channel, out_channel, same_shape=True):
        super(residual_block, self).__init__()
        self.same_shape = same_shape
        stride = 1 if self.same_shape else 2
        
        self.conv1 = conv3x3(in_channel, out_channel, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        
        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        
        if not self.same_shape:
            self.conv3 = nn.Conv2d(in_channel, out_channel, 1, stride=stride)
            # self.pool2 = nn.MaxPool2d(2)  # 前景亮度低于背景亮度时，最大池化是失败的
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(self.bn1(out))
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if not self.same_shape:
            residual = self.conv3(x)
        
        return self.relu(residual + out)


class ResNet_VAE(nn.Module):
    
    def __init__(self, input_shape):
        super(ResNet_VAE, self).__init__()
        self.input_shape = input_shape
        self.ch1, self.o_channel = 64, 128
        self.pool_kernel1 = 3
        self.fc_hidden0 = self.o_channel  # * self.pool_kernel1 * self.pool_kernel1
        
        if self.input_shape[1] > 17:
            self.CT_stride = 2
            self.fc_hidden1, self.fc_hidden2 = 512, 512
        else:
            self.CT_stride = 1
            self.fc_hidden1, self.fc_hidden2 = 128, 128
        
        # Encoder
        self.block1 = nn.Sequential(
            nn.BatchNorm2d(self.input_shape[2]),
            nn.Conv2d(in_channels=self.input_shape[2], out_channels=self.ch1, kernel_size=3, stride=1,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.ch1)
        )
        self.block21 = nn.Sequential(
            residual_block(self.ch1, self.o_channel, False),
        )
        self.block22 = nn.Sequential(
            residual_block(self.ch1, self.ch1),
            residual_block(self.ch1, self.o_channel, False),
        )
        self.block3 = nn.Sequential(
            
            residual_block(self.o_channel, self.o_channel),
            nn.AdaptiveAvgPool2d(self.pool_kernel1),
            nn.BatchNorm2d(self.o_channel, momentum=0.01),
            # nn.AvgPool2d(2),
        )
        self.block_mu = nn.Sequential(
            nn.Conv2d(self.o_channel, self.o_channel,
                      kernel_size=self.pool_kernel1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.BatchNorm1d(self.fc_hidden1, momentum=0.01),
        )
        self.block_logvar = nn.Sequential(
            nn.Conv2d(self.o_channel, self.o_channel,
                      kernel_size=self.pool_kernel1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.BatchNorm1d(self.fc_hidden1, momentum=0.01),
        )
        self.relu = nn.ReLU(inplace=True)
        self.fc11 = nn.Linear(self.fc_hidden1, self.fc_hidden2)  # 均值
        self.bn11 = nn.BatchNorm1d(self.fc_hidden2, momentum=0.01)
        self.fc12 = nn.Linear(self.fc_hidden1, self.fc_hidden2)  # 方差
        self.bn12 = nn.BatchNorm1d(self.fc_hidden2, momentum=0.01)
        # Decoder
        self.fc2 = nn.Linear(self.fc_hidden2, self.fc_hidden0)
        self.bn2 = nn.BatchNorm1d(self.fc_hidden0)
        
        self.convTrans5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.o_channel, out_channels=self.o_channel, kernel_size=5, stride=2,
                               padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.o_channel, momentum=0.01),
        )
        self.convTrans6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.o_channel, out_channels=self.ch1, kernel_size=3, stride=self.CT_stride,
                               padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.ch1, momentum=0.01),
        )
        self.convTrans7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.ch1, out_channels=self.ch1, kernel_size=3, stride=1,
                               padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.ch1, momentum=0.01),
        )
        self.convTrans8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.ch1, out_channels=self.input_shape[2], kernel_size=3,
                               stride=2,
                               padding=0),
            nn.BatchNorm2d(self.input_shape[2], momentum=0.01),
            nn.Sigmoid()  # y = (y1, y2, y3) \in [0 ,1]^3
        )
    
    def encode(self, x):
        x = self.block1(x)
        if self.CT_stride == 2:
            x = self.block22(x)
        else:
            x = self.block21(x)
        
        x = self.block3(x)
        # KL
        mu = self.block_mu(x)
        logvar = self.block_logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()  # 计算标准差
        eps = torch.randn_like(std)
        if torch.cuda.is_available():
            eps = eps.cuda()
        return eps.mul(std).add_(mu)
    
    def decode(self, z):
        z = z.view(-1, self.o_channel, 1, 1)
        x_reconstruction = self.convTrans5(z)
        x_reconstruction = self.convTrans6(x_reconstruction)
        x_reconstruction = self.convTrans7(x_reconstruction)
        x_reconstruction = self.convTrans8(x_reconstruction)
        x_reconstruction = F.interpolate(x_reconstruction, size=(self.input_shape[0], self.input_shape[1]),
                                         mode='bilinear', align_corners=False)
        return x_reconstruction
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        # z = self.reparameterize(mu, logvar)
        z = mu
        x_reconstruction = self.decode(mu)
        # z = z.view(z.shape[0], -1)
        return x_reconstruction, z, mu, logvar
