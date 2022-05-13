from __future__ import division, print_function

import os
import time
import traceback

import numpy as np
import scipy.sparse as sp
import torch
from matplotlib import pyplot as plt
from torch import optim
from torch.backends import cudnn

from Advisor_VGAE.model import GCNModelVAE
from Advisor_VGAE.optimizer import loss_function, tensor_Normalize
from Advisor_VGAE.utils import preprocess_graph
from config import config
from datautils.describe_visualise import draw, get_lr
from metrics import cal_clustering_metric

# Specify GPU, default call all
os.environ["CUDA_VISIBLE_DEVICES"] = config.g_id
Cuda = torch.cuda.is_available()  # check if GPU exists
device = torch.device("cuda" if Cuda else "cpu")

torch.set_default_tensor_type(torch.FloatTensor)


def Advisor_main(X_Inputs, T_Truths, A_ThesisFramework, Z_InitKnowledge):
    print("\n===>>> Start Advisor training:{} --->>>\n".format(str(config.save_path)))
    # X_Inputs = X_Inputs.reshape(X_Inputs.shape[0], -1)
    X_Inputs = Z_InitKnowledge
    num_nodes, dim_features = X_Inputs.shape
    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = A_ThesisFramework
    # adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    # adj_orig.eliminate_zeros()
    
    aff_y_norm = A_ThesisFramework
    
    # A_ThesisFramework 的规范化
    k, _ = torch.sort(A_ThesisFramework[2])
    A_ThesisFramework = torch.where(A_ThesisFramework > k[-50], A_ThesisFramework / k[-3], A_ThesisFramework)
    one = torch.ones_like(A_ThesisFramework)
    A_ThesisFramework = torch.where(A_ThesisFramework > 0.9, one, A_ThesisFramework)
    
    # Truth labels to Truth aff_ys
    # zero = torch.zeros_like(A_ThesisFramework)
    # eye = torch.eye(num_nodes)
    #
    # Adj_Truth = T_Truths
    # Adj_Truth_row = Adj_Truth.expand_as(A_ThesisFramework)
    # Adj_Truth_cow = Adj_Truth.unsqueeze(-1).expand_as(A_ThesisFramework)
    # Adj_Truth = torch.where(Adj_Truth_row == Adj_Truth_cow, one, zero)
    
    aff_y_norm = A_ThesisFramework
    # confidence pse-do
    adj_norm = preprocess_graph(aff_y_norm).to_dense()
    
    pos_weight = float(
        A_ThesisFramework.shape[0] * A_ThesisFramework.shape[0] - A_ThesisFramework.sum()) / A_ThesisFramework.sum()
    
    norm = A_ThesisFramework.shape[0] * A_ThesisFramework.shape[0] / float(
        (A_ThesisFramework.shape[0] * A_ThesisFramework.shape[0] - A_ThesisFramework.sum()) * 2)
    
    model = GCNModelVAE(dim_features, config.hidden1, config.hidden2, config.Advisor_dropout)
    # if Cuda:
    #     model = torch.nn.DataParallel(model).to(device)
    #     cudnn.benchmark = True
    #     model.cuda()
    #
    #     X_Inputs = X_Inputs.cuda().to(device)
    #     adj_norm = adj_norm.cuda().to(device)
    #     aff_y_norm = aff_y_norm.cuda().to(device)  # eye = eye.cuda().to(device)
    # origin ACC & NMI
    config.acc_init, config.nmi_init = cal_clustering_metric(1, T_Truths, A_ThesisFramework, X_Inputs)
    print("acc_init=", [float('{:.5f}'.format(i * 100)) for i in config.acc_init], "nmi_init=",
          [float('{:.5f}'.format(i * 100)) for i in config.nmi_init])
    with open(config.save_path + '/Advisor_epoch_result.txt', "a+") as f:
        f.write("--- Using config: {}  --- \n".format(config.save_path) + "acc_init=" + str(
            [float('{: .5f}'.format(i * 100)) for i in config.acc_init]) + "\t nmi_init=" + str(
            [float('{: .5f}'.format(i * 100)) for i in
             config.nmi_init]) + "\n================================================\n")
    
    optimizer = optim.Adam(model.parameters(), lr=config.Advisor_lr)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.Advisor_step_size, gamma=config.Advisor_gamma)
    
    plt.ion()
    plt.figure(figsize=(13, 4), dpi=100)
    for epoch in range(config.Advisor_epochs):
        config.great = False
        t = time.time()
        model.train()
        optimizer.zero_grad()
        recovered, z, mu, logvar = model(X_Inputs, adj_norm)
        
        loss = loss_function(preds=recovered, labels=aff_y_norm, mu=mu, logvar=logvar, num_nodes=num_nodes, norm=norm,
                             pos_weight=pos_weight)
        
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
        if epoch % config.iteration == 0:
            # # 清空CUDA缓存
            torch.cuda.empty_cache()
            model.eval()
            optimizer.zero_grad()
            with torch.no_grad():
                recovered, z, mu, logvar = model(X_Inputs, adj_norm)
                loss_val = loss_function(preds=recovered, labels=aff_y_norm, mu=mu, logvar=logvar, num_nodes=num_nodes,
                                         norm=norm, pos_weight=pos_weight)
                cur_loss = loss_val.item()
                hidden_emb = mu.data
                try:
                    acc_curr, nmi_curr = cal_clustering_metric(epoch, T_Truths, recovered, hidden_emb)
                    acc_curr = [x * 100 for x in acc_curr]
                    nmi_curr = [x * 100 for x in nmi_curr]
                    print("Epoch:", '%04d' % (epoch + 1), "lr=", "{:.5f}".format(get_lr(optimizer)), "t_loss=",
                          "{:.5f}".format(cur_loss), "acc_curr=", [float('{:.5f}'.format(i)) for i in acc_curr],
                          "\tnmi_curr=", [float('{:.5f}'.format(i)) for i in nmi_curr], "time=",
                          "{:.5f}".format(time.time() - t))
                    config.acc_list = np.vstack((config.acc_list, acc_curr))
                    config.nmi_list = np.vstack((config.nmi_list, nmi_curr))
                    config.loss_list.append(cur_loss)
                    draw()
                    
                    if config.great:
                        # torch.save(model, config.save_path + '/Advisor_model.pth')
                        with open(config.save_path + '/Advisor_epoch_result.txt', "a+") as f:
                            f.write(
                                "acc=" + str([float('{: .5f}'.format(i)) for i in config.best_acc]) + "\t nmi=" + str(
                                    [float('{: .5f}'.format(i)) for i in config.best_nmi]) + "\n")
                
                
                except Exception as e:
                    traceback.print_exc()
    plt.ioff()
    
    print("\n===Optimization Finished!===")
    
    print("--->>> End Advisor training:{} --->>>\n".format(str(config.save_path)))
    acc_init, nmi_init = cal_clustering_metric(1, T_Truths, aff_y_norm, X_Inputs)
    print("acc_init", "\t\t nmi_init\n", [float('{:.5f}'.format(i * 100)) for i in acc_init],
          [float('{:.5f}'.format(i * 100)) for i in nmi_init], "\nbest_acc", "\t\tbest_nmi\n",
          [float('{:.5f}'.format(i)) for i in config.best_acc], [float('{:.5f}'.format(i)) for i in config.best_nmi])
    
    draw()
