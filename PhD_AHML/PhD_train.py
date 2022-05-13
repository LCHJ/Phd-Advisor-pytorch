# -*- coding:utf-8 -*-
# @Time : 2021/3/26 10:21
# @Author: LCHJ
# @File : train.py
# coding=utf-8
# coding=utf-8
from __future__ import division

import math
import os
import random
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import optim
from torch.nn import functional as F
from torchkeras import summary
from torchvision import transforms
from tqdm import tqdm

from config import config
from datautils.describe_visualise import (contrast_show, draw, get_lr, metric_visual, plot_embeddings, )
from datautils.load_mat import mat_data
from metrics import cal_clustering_metric
from PhD_AHML.nets.siamese import AHCL, Adaptive_homotopy_model, distance

warnings.filterwarnings("ignore")

# Specify GPU, default call all
os.environ["CUDA_VISIBLE_DEVICES"] = config.g_id
Cuda = torch.cuda.is_available()  # check if GPU exists
device = torch.device("cuda" if Cuda else "cpu")


# use the same random initialization seed
# np.random.seed(0)
# torch.manual_seed(1)


# Random rotation
def variant(img_torch, a=-30, b=30):
    angle = np.random.rand() * (b - a) + a
    angle = angle * math.pi / 180
    theta = torch.tensor([[math.cos(angle), math.sin(-angle), 0], [math.sin(angle), math.cos(angle), 0]],
                         dtype=torch.float, ).to(device)
    grid = F.affine_grid(theta.unsqueeze(0), img_torch.unsqueeze(0).size()).to(device)
    output = F.grid_sample(img_torch.unsqueeze(0), grid).to(device)
    return output[0].to(device)


def train_unit(net, loss, images, targets, epoch, optimizer):
    if config.is_img_T:  # Put the picture straight
        images = images.permute(0, 1, 3, 2)
    # Random rotation image
    x = images.clone()
    ids = np.array(random.sample(range(images.size(0)), int(images.size(0) * 0.1)))
    if not config.eval:
        for i in ids:
            x[i] = variant(images[i], -45, 45)
    with torch.no_grad():
        x = x.to(device)
    # train and forward pass
    optimizer.zero_grad()  # Set the gradient to zero
    x_reconstruction, Z, mu, logvar = net(x)

    Adj = distance(Z)
    output, Y = loss(x_reconstruction, images, Adj, mu, logvar)

    if not config.eval:
        output.backward()  # Find the gradient by back propagation
        optimizer.step()  # Update all parameters
        acc, nmi = config.acc_list[-1] / 100, config.nmi_list[-1] / 100
    else:
        with torch.no_grad():  # Intermediate empty data needs to be filled when generational clustering
            acc, nmi = cal_clustering_metric(epoch, targets, Y, Z)
            if config.great:
                # show effect of refactoring
                index = ids[::10][:6]
                contrast_show(x_reconstruction[index].permute(0, 2, 3, 1), images[index].permute(0, 2, 3, 1), )

    return output.item(), acc, nmi, Y, Adj


def fit_one_epoch_mat(net, loss, epoch, epoch_size, data, label, num_train, Epoch, optimizer):
    start_time = time.time()

    data = (transforms.ToTensor()(data).view(-1, config.input_shape[2], config.input_shape[0],
                                             config.input_shape[1]).float()).to(device)
    label = torch.tensor(label).flatten().type(torch.int8).to(device)

    index = torch.argsort(label, dim=0)

    net.train()
    config.eval = False
    with tqdm(total=epoch_size, desc=f"Epoch {epoch + 1}/{Epoch}", postfix=dict, mininterval=0.5, ) as pbar:
        tqdm.write("===《Start Training》===")
        Train_loss = 0.0  # Record loss
        total_acc = np.zeros(3)  # Record acc
        total_nmi = np.zeros(3)  # Record nmi
        for iteration in range(0, epoch_size):
            # Generating random sequence
            ids = np.array(random.sample(range(num_train), min(num_train, config.Batch_size)))
            # ids = np.random.randint(0, num_train, config.Batch_size, dtype=int)
            # Organize training batch
            images = data[ids]
            targets = label[ids]
            # train
            o_loss, acc, nmi, Y, Adj = train_unit(net, loss, images, targets, epoch, optimizer)
            # result
            total_acc += acc
            total_nmi += nmi
            Train_loss += o_loss
            pbar.set_postfix(**{
                # "acc": [float("{:.5f}".format(i)) for i in (100 * total_acc)],
                # "nmi": [float("{:.5f}".format(i)) for i in (100 * total_nmi)],
                "A_mean": float("{:.5f}".format(Adj.mean())), "Y_mean": float("{:.5f}".format(Y.mean())),
                "loss"  : [float("{:.6f}".format(Train_loss / (iteration + 1)))], "lr": get_lr(optimizer), })
            pbar.update(1)

    config.loss_list.append(Train_loss / epoch_size)

    if int(epoch) % 2 == 0:
        net.eval()
        config.eval = True
        with tqdm(total=1, desc=f"Epoch {epoch + 1}/{Epoch}", postfix=dict, mininterval=0.3) as pbar:
            tqdm.write("---《Start Validating》---")
            total_acc = np.zeros(3)
            total_nmi = np.zeros(3)
            Train_loss = 0.0

            # index = np.array(random.sample(range(num_train), min(num_train, 6000)))
            if num_train > 6000:
                index = index[::2]

            X_Inputs = data[index]
            T_Truths = label[index]
            with torch.no_grad():
                o_loss, acc, nmi, Y, Adj = train_unit(net, loss, X_Inputs, T_Truths, epoch, optimizer)

                total_acc += acc
                total_nmi += nmi
                Train_loss += o_loss

            pbar.set_postfix(**{
                "acc" : [float("{:.5f}".format(i)) for i in (100 * total_acc)],
                "nmi" : [float("{:.5f}".format(i)) for i in (100 * total_nmi)],
                "loss": [float("{:.6f}".format(Train_loss))], "lr": get_lr(optimizer), })
            pbar.update(1)

            config.acc_list = np.vstack((config.acc_list, 100 * total_acc))
            config.nmi_list = np.vstack((config.nmi_list, 100 * total_nmi))
            # Visualization
            # if epoch % 10 == 0:

            plt.close()
            draw()
            if config.great:
                np.save(config.save_path + "/" + "X_Inputs.npy", X_Inputs.cpu().detach().numpy().astype(np.float32))
                plt.close()
                #     plot_embeddings(Z.cpu().detach().numpy().astype(float),
                #                     targets.cpu().detach().numpy().astype(int).flatten(), 'Z', epoch)
                #     plot_embeddings(Y.cpu().detach().numpy().astype(float),
                #                     targets.cpu().detach().numpy().astype(int).flatten(), 'Y', epoch)
                metric_visual(Y.cpu().detach().numpy().astype(float), epoch)
                # torch.save(AH_net.state_dict(), config.save_path +
                #            '/Current_good_model.pth')
                torch.save(net, config.save_path + "/Current_good_model.pth")

                try:
                    with open(config.save_path + "/Best_epoch_result.txt", "a+") as f:
                        f.write(">>>---epoch = {}---<<<\n".format(epoch) + "acc=" + str(
                            [float("{: .5f}".format(i)) for i in 100 * config.best_acc]) + "\t nmi=" + str(
                            [float("{: .5f}".format(i)) for i in 100 * config.best_nmi]) + "\n")
                except:
                    pass

            tqdm.write("\t>>-- y \t z_sp \t z_kmeans --<< " + "\nbest_acc =" + str(
                [float("{:.5f}".format(i)) for i in 100 * config.best_acc]) + "\nbest_nmi =" + str(
                [float("{:.5f}".format(i)) for i in 100 * config.best_nmi]))
    else:
        config.acc_list = np.vstack((config.acc_list, config.acc_list[-1]))
        config.nmi_list = np.vstack((config.nmi_list, config.nmi_list[-1]))

    time_elapsed = time.time() - start_time
    spend_time = time.strftime("%H:%M:%S", time.gmtime(time_elapsed))
    end_time = time.strftime("%H:%M:%S", time.gmtime(time_elapsed * (Epoch - epoch - 1)))
    tqdm.write("---> Training complete in [{}] --- Estimated time in [{}] <---".format(str(spend_time), str(end_time)))


def PhD_main():
    print("--->>> Start PhD Training:{} --->>>\n".format(str(config.save_path)))
    # load model
    AH_net = Adaptive_homotopy_model(config.input_shape)
    # -----Load pretrain model---------#
    if config.resume:
        model_path = config.save_path + "/Last_AE_model.pth"
        print("===Loading weights into state dict...===>>>{}".format(str(model_path)))
        AH_net = torch.load(model_path)

        # model_dict = AH_net.state_dict()

        # pretrained_dict = torch.load(model_path, map_location=device)  # # Remove layers with different names from the two models  # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        # model_dict.update(pretrained_dict)

        # AH_net.load_state_dict(model_dict)
    # to GPU
    if Cuda:
        # Clear CUDA cache
        torch.cuda.empty_cache()
        AH_net = torch.nn.DataParallel(AH_net).to(device)
        # AH_net = AH_net.to(device)
        cudnn.benchmark = True
    # loss
    loss = AHCL()
    # Manually set the selective gradient propagation parameters
    model_params = list(AH_net.parameters())
    for param in model_params:
        param.requires_grad = True
    optimizer = optim.Adam(AH_net.parameters(), config.D_lr, weight_decay=config.D_weight_decay)
    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
    #     optimizer, gamma=config.D_gamma, last_epoch=-1
    # )
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.D_step_size, gamma=config.D_gamma)

    # print model construction
    summary(AH_net, (config.input_shape[2], config.input_shape[0], config.input_shape[1]), )
    with open(config.save_path + "/Net_Param.txt", "w+") as f:
        f.write(str(AH_net))
    # Load dataset
    data, label, num_train = mat_data()
    # data = np_Normalize(data)
    epoch_size = math.ceil(num_train / config.Batch_size) * config.iteration_rate
    # visual:T-SNE mapping
    plot_embeddings(data, label.flatten(), "Z-Init", "0")
    config.acc_init, config.nmi_init = cal_clustering_metric(0, label, data, data)
    # 打开交互模式
    plt.ion()
    for epoch in range(config.Init_Epoch, config.Unfreeze_Epoch):
        torch.cuda.empty_cache()
        print("\n{}".format(str(config.save_path)))
        # If the type of dataset is .mat
        if config.is_mat_dat:
            fit_one_epoch_mat(AH_net, loss, epoch, epoch_size, data, label, num_train, config.Unfreeze_Epoch, optimizer)
        if epoch < config.Unfreeze_Epoch * 0.7:
            lr_scheduler.step()
    draw()
    plt.ioff()

    # Save model
    torch.save(AH_net, config.save_path + "/Last_model.pth")
    # Record the best results in the form of folder names
    os.makedirs(config.save_path + "/acc=" + str([float("{: .5f}".format(i)) for i in config.best_acc]) + "nmi=" + str(
        [float("{: .5f}".format(i)) for i in config.best_nmi]))
