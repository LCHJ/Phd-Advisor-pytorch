# -*- coding:utf-8 -*-
# @Time : 2021/3/28 14:56
# @Author: LCHJ
# @File : resnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


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
            self.bn3 = nn.BatchNorm2d(out_channel)
            # self.pool2 = nn.MaxPool2d(2)  # ǰ�����ȵ��ڱ�������ʱ�����ػ���ʧ�ܵ�
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(self.relu(out))
        out = self.conv2(out)
        out = self.bn2(out)

        if not self.same_shape:
            residual = self.bn3(self.conv3(x))

        return self.relu(residual + out)


class ResNet_VAE(nn.Module):

    def __init__(self, input_shape):
        super(ResNet_VAE, self).__init__()
        self.input_shape = input_shape
        self.ch1, self.o_channel = 64, 128
        self.pool_kernel1 = 3
        self.fc_hidden0 = self.o_channel * self.pool_kernel1 * self.pool_kernel1

        if self.input_shape[1] > 17:
            self.CT_stride = 2
            self.fc_hidden1, self.fc_hidden2 = 512, 512
        else:
            self.CT_stride = 1
            self.fc_hidden1, self.fc_hidden2 = 256, 256

        # Encoder
        self.block1 = nn.Sequential(
            nn.BatchNorm2d(self.input_shape[2]),
            nn.Conv2d(in_channels=self.input_shape[2], out_channels=self.ch1, kernel_size=3, stride=1,
                      padding=1),
            # nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.ch1)
        )
        self.block21 = nn.Sequential(

            residual_block(self.ch1, self.o_channel, False),
        )
        self.block22 = nn.Sequential(

            residual_block(self.ch1, self.ch1),
            nn.BatchNorm2d(self.ch1),
            residual_block(self.ch1, self.o_channel, False),
        )
        self.block3 = nn.Sequential(
            # nn.BatchNorm2d(self.o_channel),
            residual_block(self.o_channel, self.o_channel),
            # residual_block(self.o_channel, self.o_channel, False),
            nn.BatchNorm2d(self.o_channel),
            nn.AdaptiveAvgPool2d(self.pool_kernel1),
            # nn.AvgPool2d(2),
        )
        # Decoder
        self.block4 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.fc_hidden0, self.fc_hidden1),
            nn.BatchNorm1d(self.fc_hidden1, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.relu = nn.ReLU(inplace=True)

        self.fc11 = nn.Linear(self.fc_hidden1, self.fc_hidden2)  # ��ֵ
        self.bn11 = nn.BatchNorm1d(self.fc_hidden2, momentum=0.01)
        self.fc12 = nn.Linear(self.fc_hidden1, self.fc_hidden2)  # ����
        self.bn12 = nn.BatchNorm1d(self.fc_hidden2, momentum=0.01)
        # Decoder
        self.fc2 = nn.Linear(self.fc_hidden2, self.fc_hidden0)
        self.bn2 = nn.BatchNorm1d(self.fc_hidden0)

        self.convTrans6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.o_channel, out_channels=self.ch1, kernel_size=3, stride=2,
                               padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.ch1, momentum=0.01),
        )
        self.convTrans7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.ch1, out_channels=self.ch1, kernel_size=3, stride=self.CT_stride,
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
        # x = x.view(x.shape[0], -1)  # flatten output of conv
        x = self.block4(x)
        # FC layers
        mu = self.bn11(self.fc11(x))
        logvar = self.bn12(self.fc12(x))
        return mu, logvar

    def reparameterize(self, mu, logvar):

        std = logvar.mul(0.5).exp_()  # �����׼��
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_(0, 0.2)  # �ӱ�׼����̬�ֲ����������һ��eps
        else:
            eps = torch.FloatTensor(std.size()).normal_(0, 0.2)
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        z = self.relu(self.bn2(self.fc2(z))).view(-1, self.o_channel, self.pool_kernel1, self.pool_kernel1)

        x_reconstruction = self.convTrans6(z)
        x_reconstruction = self.convTrans7(x_reconstruction)
        x_reconstruction = self.convTrans8(x_reconstruction)
        x_reconstruction = F.interpolate(x_reconstruction, size=(self.input_shape[0], self.input_shape[1]),
                                         mode='bilinear', align_corners=False)
        return x_reconstruction

    def forward(self, x):
        mu, logvar = self.encode(x)

        z = self.reparameterize(mu, logvar)
        # z = mu
        x_reconstruction = self.decode(z)
        # z = z.view(z.shape[0], -1)
        return x_reconstruction, mu.data, mu, logvar

# def conv2D_output_size(img_size, padding, kernel_size, stride):
#     # compute output shape of conv2D
#     outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0]) / stride[0] + 1).astype(int),
#                 np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1]) / stride[1] + 1).astype(int))
#     return outshape

# def convtrans2D_output_size(img_size, padding, kernel_size, stride):
#     # compute output shape of conv2D
#     outshape = ((img_size[0] - 1) * stride[0] - 2 * padding[0] + kernel_size[0],
#                 (img_size[1] - 1) * stride[1] - 2 * padding[1] + kernel_size[1])
#     return outshape
#
#
# # ---------------------- ResNet VAE ---------------------- #
#
# class ResNet_VAE(nn.Module):
#     def __init__(self, fc_hidden1=1024, fc_hidden2=768, drop_p=0.3, CNN_embed_dim=256):
#         super(ResNet_VAE, self).__init__()
#
#         self.fc_hidden1, self.fc_hidden2, self.CNN_embed_dim = fc_hidden1, fc_hidden2, CNN_embed_dim
#         # encoding components
#         resnet18 = models.resnet18(pretrained=True)
#         self.features = nn.Sequential(
#             #
#             *(list(resnet18.children())[0:6]),
#             # (256,1,1))
#             nn.AdaptiveAvgPool2d((1, 1))
#         )
#         self.fc5 = nn.Linear(128, 64 * 2 * 2)
#         self.fc_bn5 = nn.BatchNorm1d(64 * 2 * 2)
#         self.relu = nn.ReLU(inplace=True)
