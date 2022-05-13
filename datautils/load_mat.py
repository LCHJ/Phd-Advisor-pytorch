# -*- coding:utf-8 -*-
# @Time : 2021/3/23 12:10
# @Author: LCHJ
# @File : load_mat.py
import numpy as np
from scipy import io

from config import config


def mat_data():
    # 1.read
    mat = io.loadmat(config.train_path)
    if config.data_name == 'imm40' or config.data_name == 'CIFAR_test':
        data = mat[list(mat.keys())[3]]
        # data /= np.max(np.abs(data))
        label = mat[list(mat.keys())[4]]
    elif config.data_name == 'fashion-mnist':
        data = mat['test_X']
        label = mat['test_Y']
    else:
        data = mat['X']
        label = mat['Y']
    
    # data /= np.max(np.abs(data))
    data = data / np.max(np.abs(data))
    if config.is_T:
        data = data.T
    return 1 - data, label, label.size
    # return data, label
