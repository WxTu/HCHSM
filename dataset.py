import os
from utils import *
import numpy as np
import scipy.sparse as sp
from collections import Counter
from sklearn.preprocessing import MinMaxScaler


def load(dataset):
    datadir = os.path.join('data', dataset)

    adj = np.load(f'{datadir}/adj.npy')
    diff = np.load(f'{datadir}/diff.npy')
    feat = np.load(f'{datadir}/feat.npy')
    labels = np.load(f'{datadir}/labels.npy')

    # Laplace
    feat_L = Laplace(adj, feat)

    adj = normalize_adj(adj + sp.eye(adj.shape[0])).todense()

    if dataset == "cora" or dataset == "citeseer" or dataset == "pubmed":
        train_id = np.load(f'{datadir}/idx_train.npy').squeeze(0)
        valid_id = np.load(f'{datadir}/idx_val.npy').squeeze(0)
        test_id = np.load(f'{datadir}/idx_test.npy').squeeze(0)
    else:
        train_id, valid_id, test_id = select_DatasSet(labels)

    if dataset == 'citeseer':
        feat = preprocess_features(feat)

        epsilons = [1e-5, 1e-4, 1e-3, 1e-2]
        avg_degree = np.sum(adj) / adj.shape[0]
        epsilon = epsilons[np.argmin([abs(avg_degree - np.argwhere(diff >= e).shape[0] / diff.shape[0])
                                      for e in epsilons])]

        diff[diff < epsilon] = 0.0
        scaler = MinMaxScaler()
        scaler.fit(diff)

    # feat = preprocess_features(feat)

    return adj, diff, feat, feat_L, labels, train_id, valid_id, test_id


if __name__ == '__main__':
    adj, diff, feat, feat_L, labels, train_id, valid_id, test_id = load('citeseer')
    print("adj:", type(adj), adj.shape)
    print("diff:", type(diff), diff.shape)
    print("feat:", type(feat), feat.shape)
    print("feat_L:", type(feat_L), feat_L.shape)
    print("labels:", type(labels), labels.shape)
    print("Count:", Counter(labels))
    print("train_id:", train_id.shape)
    print("valid_id:", valid_id.shape)
    print("test_id:", test_id.shape)
