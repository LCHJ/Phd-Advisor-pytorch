import pickle as pkl

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import roc_auc_score, average_precision_score


def load_data(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        '''
        fix Pickle incompatibility of numpy arrays between Python 2 and 3
        https://stackoverflow.com/questions/11305790/pickle-incompatibility-of-numpy-arrays-between-python-2-and-3
        Python的pickle模块实现了Python对象与字节流之间的二进制转换协议。即Python的pickle模块提供了Python对象的序列化/反序列化功能。
        Pickling，即序列化，特指将Python对象转换为字节流的过程。
        Unpickling，即反序列化，特指将字节流转换为Python对象的过程。
        '''
        with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as rf:
            u = pkl._Unpickler(rf)
            u.encoding = 'latin1'
            cur_data = u.load()
            objects.append(cur_data)
        # objects.append(
        #     pkl.load(open("data/ind.{}.{}".format(dataset, names[i]), 'rb')))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        "data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)
    
    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(
            min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
    
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = torch.FloatTensor(np.array(features.todense()))
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    
    return adj, features


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def mask_test_edges(adj):
    # 将图转为链路预测二分类问题
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.
    
    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0
    # 取出稀疏矩阵的上三角部分的非零元素
    adj_triu = sp.triu(adj)
    # 返回一个稀疏矩阵的非0值坐标、非0值和整个矩阵的shape, 将上三角转为tuple 返回三元组形式
    adj_tuple = sparse_to_tuple(adj_triu)
    # # 返回坐标值 coords
    edges = adj_tuple[0]
    # 这是从整个邻接矩阵得到的边的坐标
    edges_all = sparse_to_tuple(adj)[0]
    # num_test = int(np.floor(edges.shape[0] / 10.))
    # 5%数量的边作为验证集
    num_val = 10  # int(np.floor(edges.shape[0] * 0.0001))
    # edges应该是一个两位数组 每一行是一个坐标 列数就是所有边的总个数
    all_edge_idx = list(range(edges.shape[0]))
    # 通过打乱索引 来进行shuffle 而不是直接shuffle原数据
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]  # 验证集边的索引
    val_edges = edges[val_edge_idx]  # 通过索引指定对应的验证集的边
    train_edges = np.delete(edges, np.hstack([val_edge_idx]), axis=0)  # 把test和val删掉就是训练集的边
    # # ！！！ 注意 因为adj确认了没有0 所以所有的test val 和train edge都是正例！
    
    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)
        # np.all 测试沿给定轴的所有数组元素是否都计算为True
        # 这里axis=-1 就是沿着纵向找，如果这一行的元素都不为0，则返回True，否则返回False
        # np.any 测试沿给定轴的所有数组元素是否有计算为True
        # 这里axis=-1 就是沿着纵向找，如果这一行的元素有一个不为0，则返回True，否则返回False
        # 这个函数的作用是 如果坐标a是坐标集合b的其中一个，则返回True 也就是is member
    i = 0
    val_edges_false = []
    # while len(val_edges_false) < len(val_edges):
    #     idx_i = np.random.randint(0, adj.shape[0])
    #     idx_j = np.random.randint(0, adj.shape[0])
    #     if idx_i == idx_j:
    #         continue
    #     if ismember([idx_i, idx_j], train_edges):
    #         continue
    #     if ismember([idx_j, idx_i], train_edges):
    #         continue
    #     if ismember([idx_i, idx_j], val_edges):
    #         continue
    #     if ismember([idx_j, idx_i], val_edges):
    #         continue
    #     if val_edges_false:
    #         if ismember([idx_j, idx_i], np.array(val_edges_false)):
    #             continue
    #         if ismember([idx_i, idx_j], np.array(val_edges_false)):
    #             continue
    #     val_edges_false.append([idx_i, idx_j])
    #     print(i)
    #     i += 1
    # 如果上述的所有判别条件都没有让这个跳出这次循环 则把这个符合规则的样本加入到负样本集里面
    
    # ～是取反的意思
    # 以下五句话分别是确认：
    # 为测试集、验证集生成的负样本的边坐标不在所有正样本边集合里面
    # 验证集正样本和测试集正样本都不在训练集里面
    # 验证集测试集正样本木有重叠
    # 但是用上下面的五句话往往会造成内存爆炸
    # assert ~ismember(test_edges_false, edges_all)
    # assert ~ismember(val_edges_false, edges_all)
    # assert ~ismember(val_edges, train_edges)
    # assert ~ismember(test_edges, train_edges)
    # assert ~ismember(val_edges, test_edges)
    
    # data为训练集正样本个数
    data = np.ones(train_edges.shape[0])
    # Re-build adj matrix 根据之前切好的训练集正样本 重构邻接矩阵 只有训练集样本对应的位置为1
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    # 第一行的adj_train是上三角矩阵 加上转置之后的（下三角矩阵）变成完整的重构邻接矩阵
    adj_train = adj_train + adj_train.T
    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    # return sparse_to_tuple(adj_normalized)
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def get_roc_score(emb, adj_orig, edges_pos, edges_neg):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])
    
    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])
    
    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    
    return roc_score, ap_score
