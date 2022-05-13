# -*- coding:utf-8 -*-
# @Time : 2021/3/26 10:21
# @Author: LCHJ
# coding=utf-8
from __future__ import division

import torch
from torch import nn

from PhD_AHML.nets.resnet import ResNet_VAE
from config import config


class Adaptive_homotopy_model(nn.Module):

    def __init__(self, input_shape):
        super(Adaptive_homotopy_model, self).__init__()
        self.net = ResNet_VAE(input_shape)  # self.net = VGG6(input_shape[-1])

    def forward(self, x):
        # Calculate the deep embedding
        x_reconstruction, z, mu, logvar = self.net.forward(x)
        return x_reconstruction, z, mu, logvar


# Calculate Euclidean distance between nodes, namely adjacency matrix Adj
def euclidean_dist(x, Y):
    m, n = x.size(0), Y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(Y, 2).sum(1, keepdim=True).expand(n, m).T
    dist = torch.addmm(1, torch.add(xx, yy), -2, x, torch.t(Y))
    dist = dist.clamp(min=1e-8).sqrt()  # for numerical stability
    return dist


def distance(z):
    Adj = euclidean_dist(z, z)
    # normalization
    Adj = Adj / Adj.max()
    # Adj = torch.clamp(Adj / Adj[1].sort().values[-3], min=0.0, max=1.0)
    Adj = torch.max(Adj, torch.t(Adj))
    # # CosineSimilarity
    # z_norm = z / z.norm(2, dim=1).unsqueeze(dim=1)
    # C = torch.mm(z_norm, z_norm.T)
    # # ��cos����
    # C_Dist = torch.add(torch.neg(C), 1) * 0.5  # C_Dist = 1-cosine
    # C_Dist = C_Dist.clamp_(min=1e-8, max=1)  # [0,1]
    # C = torch.add(C, 1) * 0.5  # cosine = (1+cosine)/2 ,[0,1]
    # C = torch.max(C, torch.t(C))
    # C = C.clamp_(min=1e-8, max=1)  # for numerical stability
    return Adj


# MSE loss
def loss_decoder(recon_x, x, mu, logvar):
    MSE = config.Mul_MSN * nn.MSELoss(reduction='sum')(recon_x, x) / x.size(0)
    # BCE = nn.functional.binary_cross_entropy(recon_x, x) / x.size(0)
    # KLD = -0.05 * torch.mean(mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar))
    return MSE# + KLD


class AHCL(torch.nn.Module):

    def __init__(self):
        super(AHCL, self).__init__()
        self.margin = config.margin

    def forward(self, x_reconstruction, x, Adj, mu, logvar):
        f = torch.pow(Adj, 2)
        # D = Adj.clamp(min=1e-8).sqrt()
        g = torch.pow(torch.clamp(self.margin - Adj, min=1e-8), 2)

        w = torch.div(f, torch.add(f, g))
        with torch.no_grad():
            Y = torch.div(g, torch.add(f, g))
            Y = torch.max(Y, torch.t(Y))

            # k , _ =torch.sort(Y[2])  # Y = torch.where(Y > k[-50], Y / k[-2], Y)  # one = torch.ones_like(Y).cuda()  # Y = torch.where(Y > 0.9, one, Y)

        loss_contrastive = config.Multiple_AHML * torch.mean(
            torch.mul(Y, f) + torch.mul(1 - Y, g)) + 0.00001 * torch.norm(w, 2)
        loss_VAE = loss_decoder(x_reconstruction, x, mu, logvar)
        return loss_contrastive + loss_VAE, Y


def np_Standardization(data):
    return (data - data.mean()) / data.std()


def np_Normalize(data):
    return (data - data.min()) / (data.max() - data.min())


def tensor_Standardization(data):
    return (data - data.mean()) / data.std()


def tensor_Normalize(data):
    return (data - data.min()) / (data.max() - data.min())
