# -*- coding:utf-8 -*-
# @Time : 2021/1/22 21:48
# @Author: LCHJ
# @File : describe_visualise.py
# coding=utf-8


import os
import sys

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.manifold import TSNE

from config import config

sys.path.append("../")

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 更新字体格式


# 统计所有类图片的总数
def get_image_num(path):
    num = 0
    train_path = os.path.join(path)
    for character in os.listdir(train_path):
        # 在大众类下遍历小种类。
        character_path = os.path.join(train_path, character)
        num += len(os.listdir(character_path))
    return num


# 获取当前学习率lr
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


# rgb to gray
def rgb2gray(rgb):
    return (0.2989 * rgb[:, 0] + 0.5870 * rgb[:, 1] + 0.1140 * rgb[:, 2]).view(-1, 1, config.input_shape[0],
                                                                               config.input_shape[1])


def draw():
    # 保存数据深层表征为.npy
    np.save(config.save_path + "/" + config.data_name + "_Acc_vs_epoch.npy", config.acc_list)
    np.save(config.save_path + "/" + config.data_name + "_NMI_vs_epoch.npy", config.nmi_list)
    np.save(config.save_path + "/" + config.data_name + "_loss_vs_epoch.npy", config.loss_list)
    # show
    x1 = range(0, len(config.acc_list))
    x2 = range(1, len(config.loss_list) + 1)
    Title = ["ACC vs. epoch", "NMI vs. epoch", "Loss vs. epoch"]
    legend_list = ['y_sp', 'z_sp', 'z_km', 'z_sp_o', 'z_km_o']
    plt.close()
    plt.figure(figsize=(13, 4), dpi=100,)
    plt.suptitle(config.save_path[7:])
    plt.tight_layout()  # 调整整体空白
    plt.subplots_adjust(hspace=0, wspace=0.2)
    ax = plt.subplot(1, 3, 1)

    ax.plot(x1, config.acc_list[:, 0], "g.-", label=legend_list[0], markersize=3)
    ax.plot(x1, config.acc_list[:, 1], "b.-", label=legend_list[1], markersize=3)
    ax.plot(x1, config.acc_list[:, 2], "c.-", label=legend_list[2], markersize=3)

    ax.axhline(100 * config.acc_init[1], color='g', linestyle='-.', linewidth=0.5, alpha=0.5, label='legend_list[2]')
    ax.axhline(100 * config.acc_init[2], color='c', linestyle='--', linewidth=0.5, alpha=0.5, label='legend_list[3]')

    ax.legend(legend_list, loc="lower right")
    ax.set_title(Title[0], fontsize="small")
    ax.set_xlabel(str([float("{:.4f}".format(i)) for i in 100 * config.best_acc]))
    ax.set_ylabel("ACC")

    ax2 = plt.subplot(1, 3, 2)

    ax2.plot(x1, config.nmi_list[:, 0], "g.-", label=legend_list[0], markersize=3)
    ax2.plot(x1, config.nmi_list[:, 1], "b.-", label=legend_list[1], markersize=3)
    ax2.plot(x1, config.nmi_list[:, 2], "c.-", label=legend_list[2], markersize=3)

    ax2.axhline(100 * config.nmi_init[1], color='g', linestyle='-.', linewidth=0.5, alpha=0.5, label='legend_list[2]')
    ax2.axhline(100 * config.nmi_init[2], color='c', linestyle='--', linewidth=0.5, alpha=0.5, label='legend_list[3]')

    ax2.legend(legend_list, loc="lower right")
    ax2.set_title(Title[1], fontsize="small")
    ax2.set_xlabel(str([float("{:.4f}".format(i)) for i in 100 * config.best_nmi]))
    ax2.set_ylabel("NMI")

    # loss
    ax3 = plt.subplot(1, 3, 3)
    ax3.plot(x2, config.loss_list, "r.-", label="loss", markersize=3)
    ax3.legend(["loss"], loc="upper right")
    ax3.set_title(Title[2], fontsize="small")
    ax3.set_xlabel(str("{:.5f}".format(config.loss_list[-1])))
    ax3.set_ylabel("Loss")

    plt.savefig(config.save_path + "/" + config.data_name + "_Acc&loss_vs_epoch.png")
    plt.pause(0.5)


# 原数据和生成数据的比较
def contrast_show(x_reconstruction, x):
    x_reconstruction = x_reconstruction.cpu().detach().numpy()
    x = x.cpu().detach().numpy()
    # 保存为npy
    np.save(config.save_path + "/" + "reconstruction.npy", x_reconstruction)
    np.save(config.save_path + "/" + "Origin.npy", x)
    # show
    # plt.clf()
    plt.close()
    fig, ax = plt.subplots(2, len(x), sharex="col", sharey="row", figsize=(6, 2.6), dpi=160)
    fig.suptitle(config.save_path[7:], fontsize="small")
    fig.tight_layout()  # 调整整体空白
    fig.subplots_adjust(hspace=0, wspace=0)
    for i in range(len(x)):
        ax[0, i].axis("off")
        ax[1, i].axis("off")
        ax[0, i].imshow(x[i], cmap="Greys")
        ax[1, i].imshow(x_reconstruction[i], cmap="Greys")

    plt.savefig(config.save_path + "/" + config.data_name + "_Origin_vs_Reconstruct.png")
    plt.pause(1)


#  visualization


def plot_embeddings(embeddings, Labels, kind="Z", epoch="0", Is_good=False):
    print("\n Computing t-SNE embedding-" + str(kind) + str(epoch) + "\n")
    emb_list = np.array(embeddings)
    model = TSNE(n_components=2, init="random", perplexity=30, early_exaggeration=12)
    node_pos = model.fit_transform(emb_list)

    plt.close()
    plt.figure(figsize=(8, 6), dpi=100)
    plt.title(config.data_name + "-{}_{}".format(kind, epoch))
    plt.axis("off")

    # color
    color_idx = {}
    for i in range(embeddings.shape[0]):
        color_idx.setdefault(Labels[i], [])
        color_idx[Labels[i]].append(i)
    # marker
    markers = [".", "o", "s", "1", "2", "^", "v", ">", "<", "D"]

    # plot
    for c, idx in color_idx.items():
        # 根据权重
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c, s=10, marker=markers[int(c) % 10], alpha=1, )

    plt.savefig(config.save_path + "/" + str("000{}_{}_TSNE.png".format(epoch, kind))[-17:])
    plt.pause(0.5)
    return node_pos, color_idx


def metric_visual(Y, epoch="0"):
    # modified to 0 or 1 closely
    y = np.sort(Y[100])
    print("{}\n{}\n".format(y[:5], y[-5:]))

    Y = np.where(Y > y[-300], Y / y[-3], Y)
    # Y = np.where(Y > 0.02, Y*80.0, Y)

    img = np.where(Y > 0.9, 1.0, Y)

    plt.close()
    # plt.figure(figsize=(8, 6), dpi=100)
    fig = plt.figure()
    fig.set_size_inches(8, 6, forward=True)
    # 作图并选择热图的颜色填充风格，这里选择hot
    im = plt.imshow(img, cmap=plt.cm.hot_r)
    # 增加右侧的颜色刻度条
    plt.colorbar(im)
    # 增加标题
    plt.title(config.data_name + "-{}_Y".format(epoch))
    plt.savefig(config.save_path + "/" + str("000{}_Y.png".format(epoch))[-10:])
    plt.pause(0.5)


def gen_visual(epoch):
    Z = np.load(config.save_path + "/" + str("000{}_Z.npy".format(epoch))[-10:])
    T = np.load(config.save_path + "/" + str("000{}_T.npy".format(epoch))[-10:])
    Y = np.load(config.save_path + "/" + str("000{}_Y.npy".format(epoch))[-10:])

    sorted_indices = np.argsort(T)
    T = T[sorted_indices]
    Z = Z[sorted_indices]
    Y = Y[:, sorted_indices][sorted_indices, :]

    # Y = torch.from_numpy(Y).float()
    # T = torch.from_numpy(T).float()
    # ground truth of Y
    # zero = torch.zeros_like(Y)
    # one = torch.ones_like(Y)
    # o_labels_row = T.expand_as(Y)
    # o_labels_cow = T.unsqueeze(-1).expand_as(Y)
    # T = torch.where(o_labels_row == o_labels_cow, one, zero)
    # Y = torch.where(T == 0, Y.pow(2), Y.sqrt())
    # Y = torch.where(Y > 1.0, one, Y)
    # Y = Y.cpu().detach().numpy().astype(float)
    # np.save(config.save_path + "/" + str("000{}_Y.npy".format(epoch))[-10:], Y)

    metric_visual(Y, epoch)
    plot_embeddings(Y, T, "Y", epoch)
    return Z, Y, T


def makeplot():
    # visual for Y & Z
    plt.ion()
    for skip in range(0, 100):
        kk = 3
        epoch = skip * kk
        try:
            print("--->>> epoch = {}".format(epoch))
            gen_visual(epoch)
        except:
            print("--->>> <pass[{}]> --->>>".format(epoch))
            pass
    plt.ioff

    # plot loss  # name = ['UMIST', 'COIL20', 'Palm', 'Fashion', 'USPS']  # path = [r".\logs\umist_1024\AE-mix-KL0-0.7_1.0_0.008_1.0", r".\logs\COIL20\AE-mix-KL0-0.7_1.0_0.008_1.0_ok",  #         r".\logs\Palm\AE-mix-00KL0.64_1.0_0.01_1.0",  #         r".\logs\fashion-mnist\512-512-AE-max-0AE0.76_1.0_0.008_1.0_ok",  #         r".\logs\USPS\AE128-mix-00KL0.7_1.0_0.008_1.0_ok"]  # color_idx = ['r', 'm', 'b', 'g', 'c']  #  # for i in range(0, len(path)):  #     temp = np.load(path[i] + '/' + 'loss_vs_epoch.npy')[0:200]  #     draw_loss(temp, color_idx[i], name[i])


def draw_loss(loss, color, name):
    # show
    x1 = range(1, len(loss) + 1)
    plt.gcf().set_size_inches(512 / 100, 512 / 100)
    plt.subplots_adjust(top=1, bottom=0.1, right=0.9, left=0.0, hspace=0, wspace=0)
    plt.margins(0.03, 0.05)
    # plt.axis('off')
    ax = plt.subplot(111)
    ax.plot(x1, loss, color=color, linestyle="-", marker="o", label=name, markersize=5)
    ax.legend([name[i]], loc="upper right", fontsize=23)
    # ax.set_title(Title, fontsize='small')
    # 设置横纵坐标的名称以及对应字体格式
    ax.set_xlabel("epoch", fontsize=28)
    ax.set_ylabel("loss", fontsize=28)
    # MultipleLocator()函数设置了x轴相邻显示点的间隔
    x_major_locator = MultipleLocator(50)
    ax.xaxis.set_major_locator(x_major_locator)
    plt.xticks(fontsize=23)
    plt.yticks(fontsize=23)
    plt.savefig(r"C:\Users\HP\OneDrive\paper\NeurIPS 2021" + "/" + str(name) + "_loss_vs_epoch.png",
                bbox_inches="tight", )
    plt.show()
