import torch
import torch.nn.functional as F
import torch.nn.modules.loss
from torch import nn


def tensor_Standardization(data):
    return (data - data.mean(0)[0]) / data.std(0)[0]


def tensor_Normalize(data):
    return (data - data.min(0)[0]) / (data.max(0)[0] - data.min(0)[0])


def loss_function(preds, labels, mu, logvar, num_nodes, norm, pos_weight):
    # cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)
    preds = preds / preds.max()
    MSE = nn.MSELoss(reduction='sum')(preds, labels) / num_nodes
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.05 * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return MSE + KLD
