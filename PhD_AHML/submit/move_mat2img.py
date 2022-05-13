# -*- coding:utf-8 -*-
# @Time : 2020/12/30 11:07
# @Author: LCHJ
# @File : savemnist.py.py

'''将二进制格式的MNIST数据集转成.jpg图片格式并保存，图片标签包含在图片名中'''
import os

import cv2
import numpy as np


def save_mnist_to_jpg(mnist_image_file, mnist_label_file, save_dir):
    if 'train' in os.path.basename(mnist_image_file):
        num_file = 60000
        prefix = 'train'
    else:
        num_file = 10000
        prefix = '0.7_1_0.008_0.8mix'
    with open(mnist_image_file, 'rb') as f1:
        image_file = f1.read()
    with open(mnist_label_file, 'rb') as f2:
        label_file = f2.read()
    image_file = image_file[16:]
    label_file = label_file[8:]
    # mnist共10类
    for i in range(0, 10):
        path = os.path.join(save_dir, str(i))
        if not os.path.exists(path):
            os.makedirs(path)
    
    for i in range(num_file):
        label = int(label_file[i])
        image_list = [int(item) for item in image_file[i * 784:i * 784 + 784]]
        image_np = np.array(image_list, dtype=np.uint8).reshape(28, 28, 1)
        save_name = os.path.join(save_dir + "/" + str(label) + "/", '{}_{}_{}.jpg'.format(prefix, i, label))
        cv2.imwrite(save_name, image_np)
        print('{} ==> {}_{}_{}.jpg'.format(i, prefix, i, label))


if __name__ == '__main__':
    train_image_file = './datasets/train-images.idx3-ubyte'
    train_label_file = './datasets/train-labels.idx1-ubyte'
    test_image_file = 'datasets/t10k-images.idx3-ubyte'
    test_label_file = './datasets/t10k-labels.idx1-ubyte'
    
    save_train_dir = './datasets/mnist/mnist60000'
    save_test_dir = './datasets/mnist/mnist10000'
    
    if not os.path.exists(save_train_dir):
        os.makedirs(save_train_dir)
    if not os.path.exists(save_test_dir):
        os.makedirs(save_test_dir)
    
    save_mnist_to_jpg(train_image_file, train_label_file, save_train_dir)
    save_mnist_to_jpg(test_image_file, test_label_file, save_test_dir)
