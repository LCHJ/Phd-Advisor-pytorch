# -*- coding:utf-8 -*-
# @Time : 2021/3/28 14:50
# @Author: LCHJ
# @File : vgg.py
import torch.nn as nn


class VGG(nn.Module):
    
    def __init__(self, features, num_classes=128):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((5, 5))
        self.classifier = nn.Sequential(
            nn.Linear(128 * 5 * 5, num_classes),
            # nn.ReLU(True),
            # nn.Dropout(p=0.5),
            # nn.Linear(2048, num_classes),
        )
        self._initialize_weights()
    
    def forward(self, x):
        x = self.features(x)
        # x = self.avgpool(x)
        # 在第一维上将每个image拉成一维
        x = x.view(x.size()[0], -1)
        # x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # 是否为批归一化层
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False, in_channels=1):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


'''
1、一张原始图片被resize到指定大小，本文使用105x105。
2、conv1包括两次[3,3]卷积网络，一次2X2最大池化，输出的特征层为64通道。
3、conv2包括两次[3,3]卷积网络，一次2X2最大池化，输出的特征层为128通道。
4、conv3包括三次[3,3]卷积网络，一次2X2最大池化，输出的特征层为256通道。
5、conv4包括三次[3,3]卷积网络，一次2X2最大池化，输出的特征层为512通道。
6、conv5包括三次[3,3]卷积网络，一次2X2最大池化，输出的特征层为512通道。
'''
cfgs = {
    # 'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    'D': [64, 64, 'M', 128, 128]
}


def VGG6(in_channels, **kwargs):
    model = VGG(make_layers(cfgs["D"], batch_norm=True, in_channels=in_channels), **kwargs)
    return model
