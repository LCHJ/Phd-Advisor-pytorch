# -*- coding:utf-8 -*-
# @Time : 2021/7/26 10:34
# @Author: LCHJ
# @File : Phd_train.py
# coding=utf-8

from __future__ import division

import os
import warnings

import numpy as np
import torch

from Advisor_VGAE.Advisor_train import Advisor_main

from PhD_AHML.PhD_train import PhD_main
from PhD_AHML.Phd_test import Phd_test
from config import config

warnings.filterwarnings('ignore')

# Use the same random initialization seed to create the same random effect
# np.random.seed(6)
# torch.manual_seed(8)

# Specify GPU, default call all
os.environ["CUDA_VISIBLE_DEVICES"] = config.g_id
Cuda = torch.cuda.is_available()  # check if GPU exists
device = torch.device("cuda" if Cuda else "cpu")


def load_npy():
    X_Inputs = np.load(config.save_path + "/" + 'X_Inputs.npy')
    T_Truths = np.load(config.save_path + "/" + 'T_Truths.npy')
    A_ThesisFramework = np.load(config.save_path + "/" + 'A_ThesisFramework.npy')
    Z_InitKnowledge = np.load(config.save_path + "/" + 'Z_InitKnowledge.npy')

    # To tensor
    X_Inputs = torch.from_numpy(X_Inputs).float()
    T_Truths = torch.from_numpy(T_Truths).float().int()
    A_ThesisFramework = torch.from_numpy(A_ThesisFramework).float()
    Z_InitKnowledge = torch.from_numpy(Z_InitKnowledge).float()
    return X_Inputs, T_Truths, A_ThesisFramework, Z_InitKnowledge


if __name__ == '__main__':
    # make dir
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    print("===>>> Start:{} <<<===\n".format(str(config.save_path)))

    config.Phd_train = False
    config.Phd_test = True
    config.Advisor_train = True

    if config.Phd_train:
        PhD_main()

    if config.Phd_test:
        X_Inputs, T_Truths, A_ThesisFramework, Z_InitKnowledge = Phd_test()
    else:
        X_Inputs, T_Truths, A_ThesisFramework, Z_InitKnowledge = load_npy()

    if config.Advisor_train:
        '''
        # Learned the draft generated and Thesis Framework from Phd,
        # Use the matrix A_ThesisFramework as the adjacency matrix for Advisor, and X_Inputs for Inputs
        '''
        config.save_path = config.Advisor_path
        if not os.path.exists(config.save_path):
            os.makedirs(config.save_path)
        Advisor_main(X_Inputs, T_Truths, A_ThesisFramework, Z_InitKnowledge)
