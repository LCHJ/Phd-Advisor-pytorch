# -*- coding:utf-8 -*-
# @Time : 2021/3/26 10:21
# @Author: LCHJ
# @File : train.py
# coding=utf-8
# coding=utf-8
from __future__ import division

import os
import time
import warnings

import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from config import config
from datautils.load_mat import mat_data
from metrics import cal_clustering_metric
from PhD_AHML.nets.siamese import AHCL, distance, np_Normalize

warnings.filterwarnings("ignore")

# gpu or cpu
# Specify GPU, default call all
os.environ["CUDA_VISIBLE_DEVICES"] = config.g_id
Cuda = torch.cuda.is_available()  # check if GPU exists
device = torch.device("cuda" if Cuda else "cpu")


# 为了使用同样的随机初始化种子以形成相同的随机效果
# np.random.seed(0)
# torch.manual_seed(1)


def evaluate(net, loss, data, label, num_train):
    start_time = time.time()
    
    data = (
        transforms.ToTensor()(data)
            .view(-1, config.input_shape[2], config.input_shape[0], config.input_shape[1])
            .float()
    )
    label = torch.tensor(label).flatten().type(torch.int8)
    
    data = data.to(device)
    label = label.to(device)
    
    net.eval()
    with tqdm(total=1, desc=f"1 {0 + 1}/{1}", postfix=dict, mininterval=0.3) as pbar:
        tqdm.write("---《Start Validating》---")
        total_acc = np.zeros(3)
        total_nmi = np.zeros(3)
        Train_loss = 0.0
        
        # index = torch.argsort(label, dim=0)
        X_Inputs = data[: min(num_train, 10000)]
        T_Truths = label[: min(num_train, 10000)]
        
        # index = torch.argsort(T_Truths, dim=0)
        # X_Inputs = X_Inputs[index]
        # T_Truths = T_Truths[index]
        with torch.no_grad():
            x_reconstruction, Z_InitKnowledge, mu, logvar = net(X_Inputs)  # train and forward pass
            A = distance(Z_InitKnowledge)
            output, A_ThesisFramework = loss(x_reconstruction, X_Inputs, A, mu, logvar)
        
        acc, nmi = cal_clustering_metric(0, T_Truths, A_ThesisFramework, Z_InitKnowledge)
        
        total_acc += acc
        total_nmi += nmi
        Train_loss += output.item()
        pbar.set_postfix(
            **{
                "acc" : [float("{:.5f}".format(i)) for i in (100 * total_acc)],
                "nmi" : [float("{:.5f}".format(i)) for i in (100 * total_nmi)],
                "loss": Train_loss,
            }
        )
        pbar.update(1)
        
        # Visualization
        # plt.ion()
        # plot_embeddings(Z_InitKnowledge.cpu().detach().numpy().astype(float),
        #                 T_Truths.cpu().detach().numpy().astype(int).flatten(), 'Z_InitKnowledge', 0)
        # plot_embeddings(A_ThesisFramework.cpu().detach().numpy().astype(float),
        #                 T_Truths.cpu().detach().numpy().astype(int).flatten(), 'A_ThesisFramework', 0)
        # metric_visual(A_ThesisFramework.cpu().detach().numpy().astype(float), 0)
        # plt.ioff()
        
        tqdm.write(
            "\\t*-- y \t  z_sp  \t z_kmeans--* "
            + "\nbest_acc ="
            + str([float("{:.5f}".format(i)) for i in (100 * total_acc)])
            + "\nbest_nmi ="
            + str([float("{:.5f}".format(i)) for i in (100 * total_nmi)])
        )
        
        spend_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        tqdm.write("---Evaluate complete in :" + str(spend_time))
        
        return X_Inputs.cpu(), T_Truths.cpu(), A_ThesisFramework.cpu(), Z_InitKnowledge.cpu()


def Phd_test():
    print("--->>> Start PhD Testing:{} --->>>\n".format(str(config.save_path)))
    
    # -----Load pretrain model---------#
    model_path = config.save_path + "/Current_good_model.pth"
    # model_path = config.save_path +'/Last_model.pth'
    print("===Loading weights into state dict...===>>>{}".format(str(model_path)))
    AH_net = torch.load(model_path)
    
    # to GPU
    if Cuda:
        # Empty CUDA Cache
        # torch.cuda.empty_cache()
        AH_net = AH_net.to(device)
        # cudnn.benchmark = True
    
    loss = AHCL()
    # Load dataset
    data, label, num_train = mat_data()
    # data = np_Normalize(data)
    # visual:T-SNE mapping
    # plot_embeddings(data, label.flatten(), 'Z_InitKnowledge', 'O')
    X_Inputs, T_Truths, A_ThesisFramework, Z_InitKnowledge = evaluate(AH_net, loss, data, label, num_train)
    return X_Inputs, T_Truths, A_ThesisFramework, Z_InitKnowledge
