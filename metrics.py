import numpy as np
import torch
from munkres import Munkres
from sklearn import metrics
from sklearn.cluster import spectral_clustering, SpectralClustering, KMeans
from sklearn.metrics import roc_auc_score, average_precision_score

from config import config


class linkpred_metrics:

    def __init__(self, edges_pos, edges_neg):
        self.edges_pos = edges_pos
        self.edges_neg = edges_neg

    def get_roc_score(self, emb, feas):
        # if emb is None:
        #     feed_dict.update({placeholders["dropout"]: 0})
        #     emb = sess.run(model.z_mean, feed_dict=feed_dict)

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        # Predict on 0.7_1_0.008_0.8mix set of edges
        adj_rec = np.dot(emb, emb.T)
        preds = []
        pos = []
        for e in self.edges_pos:
            preds.append(sigmoid(adj_rec[e[0], e[1]]))
            pos.append(feas["adj_orig"][e[0], e[1]])

        preds_neg = []
        neg = []
        for e in self.edges_neg:
            preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
            neg.append(feas["adj_orig"][e[0], e[1]])

        preds_all = np.hstack([preds, preds_neg])
        labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
        roc_score = roc_auc_score(labels_all, preds_all)
        ap_score = average_precision_score(labels_all, preds_all)

        return roc_score, ap_score, emb


class ClusteringMetrics:

    def __init__(self, true_label, predict_label):
        self.true_label = true_label
        self.pred_label = predict_label

    def clusteringAcc(self):
        # best mapping between true_label and predict label
        l1 = list(set(self.true_label))
        numclass1 = len(l1)

        l2 = list(set(self.pred_label))
        numclass2 = len(l2)
        if numclass1 != numclass2:
            print("Class Not equal, Error!!!!")
            return 0

        cost = np.zeros((numclass1, numclass2), dtype=int)
        for i, c1 in enumerate(l1):
            mps = [i1 for i1, e1 in enumerate(self.true_label) if e1 == c1]
            for j, c2 in enumerate(l2):
                mps_d = [i1 for i1 in mps if self.pred_label[i1] == c2]

                cost[i][j] = len(mps_d)

        # match two clustering results by Munkres algorithm
        m = Munkres()
        cost = cost.__neg__().tolist()

        indexes = m.compute(cost)

        # get the match results
        new_predict = np.zeros(len(self.pred_label))
        for i, c in enumerate(l1):
            # correponding label in l2:
            c2 = l2[indexes[i][1]]

            # ai is the index with label==c2 in the pred_label list
            ai = [ind for ind, elm in enumerate(self.pred_label) if elm == c2]
            new_predict[ai] = c

        acc = metrics.accuracy_score(self.true_label, new_predict)
        f1_macro = metrics.f1_score(self.true_label, new_predict, average="macro")
        precision_macro = metrics.precision_score(self.true_label, new_predict, average="macro")
        recall_macro = metrics.recall_score(self.true_label, new_predict, average="macro")
        f1_micro = metrics.f1_score(self.true_label, new_predict, average="micro")
        precision_micro = metrics.precision_score(self.true_label, new_predict, average="micro")
        recall_micro = metrics.recall_score(self.true_label, new_predict, average="micro")
        return (acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro,)

    def evaluationClusterModelFromLabel(self, print_msg=True):
        nmi = metrics.normalized_mutual_info_score(self.true_label, self.pred_label)
        adjscore = metrics.adjusted_rand_score(self.true_label, self.pred_label)
        (acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro,) = self.clusteringAcc()

        if print_msg:
            print(
                "\nACC=%f, f1_macro=%f, precision_macro=%f, recall_macro=%f, f1_micro=%f, precision_micro=%f, recall_micro=%f, NMI=%f, ADJ_RAND_SCORE=%f" % (
                    acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, nmi,
                    adjscore,))

            # fh = open("recoder.txt", "a")  #  # fh.write("ACC=%f, f1_macro=%f, precision_macro=%f, recall_macro=%f, f1_micro=%f, precision_micro=%f, recall_micro=%f, NMI=%f, ADJ_RAND_SCORE=%f" % (acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, nmi, adjscore) )  # fh.write("\r\n")  # fh.flush()  # fh.close()

        return acc, nmi


def cal_clustering_acc(true_label, pred_label):
    l1 = list(set(true_label))
    numclass1 = len(l1)

    l2 = list(set(pred_label))
    numclass2 = len(l2)
    if numclass1 != numclass2:
        print("Class Not equal, Error!!!!")
        return 0

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(true_label) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if pred_label[i1] == c2]

            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()

    indexes = m.compute(cost)

    # get the match results
    new_predict = np.zeros(len(pred_label))
    for i, c in enumerate(l1):
        # correponding label in l2:
        c2 = l2[indexes[i][1]]

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(pred_label) if elm == c2]
        new_predict[ai] = c

    acc = metrics.accuracy_score(true_label, new_predict)
    # acc = metrics.precision_score(true_label, new_predict, average="macro")  # Micro average, precision rate
    return acc


def gpu_2_cpu(truth, Y, Z):
    # 保存数据深层表征为.npy
    truth = truth.cpu().detach().numpy().astype(int).flatten() if torch.is_tensor(truth) else np.asarray(truth).astype(int).flatten()
    Y = Y.cpu().detach().numpy().astype(np.float32) if torch.is_tensor(Y) else np.asarray(Y).astype(np.float32)
    Z = Z.cpu().detach().numpy().astype(np.float32) if torch.is_tensor(Z) else np.asarray(Z).astype(np.float32)
    return truth, Y, Z


def clustering(Y, Z):
    z_labels = SpectralClustering(n_clusters=config.num_classes, affinity="nearest_neighbors").fit_predict(Z)
    z_kmeans = KMeans(n_clusters=config.num_classes).fit_predict(Z)
    try:
        y_labels = spectral_clustering(Y, n_clusters=config.num_classes)
    except:
        y_labels = KMeans(n_clusters=config.num_classes).fit_predict(Y)

    labels = [y_labels, z_labels, z_kmeans]
    return labels


def cal_clustering_metric(epoch, truth, Y, Z):
    truth, Y, Z = gpu_2_cpu(truth, Y, Z)
    acc = []
    nmi = []
    prediction = clustering(Y, Z)
    for i in range(0, len(prediction)):
        acc.append(cal_clustering_acc(truth, prediction[i]))
        nmi.append(metrics.normalized_mutual_info_score(truth, prediction[
            i]))  # acc0, nmi0 = ClusteringMetrics(truth, prediction[i])

    config.great = False
    for i in range(0, len(config.best_acc)):
        if config.best_acc[i] < acc[i] and config.best_nmi[i] < nmi[i]:
            config.best_acc[i] = acc[i]
            config.best_nmi[i] = nmi[i]
            config.great = True

    if config.great:
        # np.save(config.save_path + "/" + str("000{}_T.npy".format(epoch))[-10:], truth, )
        # np.save(config.save_path + "/" + str("000{}_Y.npy".format(epoch))[-10:], Y)
        # np.save(config.save_path + "/" + str("000{}_Z.npy".format(epoch))[-10:], Z)
        np.save(config.save_path + "/" + "T_Truths.npy", truth)
        np.save(config.save_path + "/" + "A_ThesisFramework.npy", Y)
        np.save(config.save_path + "/" + "Z_InitKnowledge.npy", Z)

    return acc, nmi
