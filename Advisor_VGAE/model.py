# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from Advisor_VGAE.layers import GraphConvolution


class GCNModelVAE(nn.Module):
    
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelVAE, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)
    
    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        return self.dc(z), z, mu, logvar


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""
    
    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act
    
    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj
        
        # Dist = euclidean_dist(z, z)
        # Dist = Dist / Dist.max()
        #
        # f = torch.pow(Dist, 2)
        # g = torch.pow(torch.clamp(0.8 - Dist, min=0), 2)
        # return torch.div(g, torch.add(f, g))


# 计算节点间欧式距离，即邻接矩阵A
def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).T
    dist = torch.addmm(1, torch.add(xx, yy), -2, x, torch.t(y))
    dist = dist.clamp(min=1e-8).sqrt()  # for numerical stability
    return dist
