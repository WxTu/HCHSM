import torch
import numpy as np
import torch.nn as nn
from dataset import load
import scipy.sparse as sp
from HCHSM import Model, LogReg
from utils import sparse_mx_to_torch_sparse_tensor


def Train(dataset="citeseer", nb_epochs=1000, patience=40, lr=0.001, l2_coef=0.0, hid_units=512,
          sample_size=2000, batch_size=1, small_ratio_=1, cnt_wait=0, best=1e9,
          best_t=0, gamma=0.1, sigma=0.01, drop_rate=0.1, cuda=True, sparse=False, verbose=False):
    adj, diff, features, feat_L, labels, idx_train, idx_val, idx_test = load(dataset)

    ft_size = features.shape[1]
    nb_classes = np.unique(labels).shape[0]

    labels = torch.LongTensor(labels)
    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)

    lbl_1 = torch.ones(batch_size, sample_size)
    lbl_2 = torch.zeros(batch_size, sample_size)
    lbl = torch.cat((lbl_1, lbl_2), 1)

    model = Model(ft_size, hid_units, drop_rate)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

    if cuda:
        model.cuda()
        labels = labels.cuda()
        lbl = lbl.cuda()
        idx_train = idx_train.cuda()
        idx_test = idx_test.cuda()

    b_xent = nn.BCEWithLogitsLoss()
    xent = nn.CrossEntropyLoss()

    for epoch in range(nb_epochs):

        small_ratio = small_ratio_ * (1 - epoch / nb_epochs)
        if small_ratio <= 0.1:
            small_ratio = 0.1

        sub_sample_size = int(small_ratio * sample_size)
        lbl_3 = torch.ones(batch_size, sub_sample_size)
        lbl_4 = torch.zeros(batch_size, sub_sample_size)
        lbl1 = torch.cat((lbl_3, lbl_4), 1)

        sub_sub_sample_size = int(small_ratio * int(small_ratio * sample_size))
        lbl_5 = torch.ones(batch_size, sub_sub_sample_size)
        lbl_6 = torch.zeros(batch_size, sub_sub_sample_size)
        lbl2 = torch.cat((lbl_5, lbl_6), 1)

        if cuda:
            lbl1 = lbl1.cuda()
            lbl2 = lbl2.cuda()

        idx = np.random.randint(0, adj.shape[-1] - sample_size + 1, batch_size)
        ba, bd, bf, bl = [], [], [], []
        for i in idx:
            ba.append(adj[i: i + sample_size, i: i + sample_size])
            bd.append(diff[i: i + sample_size, i: i + sample_size])
            bf.append(features[i: i + sample_size])
            bl.append(feat_L[i: i + sample_size])

        ba = np.array(ba).reshape(batch_size, sample_size, sample_size)
        bd = np.array(bd).reshape(batch_size, sample_size, sample_size)
        bf = np.array(bf).reshape(batch_size, sample_size, ft_size)
        bl = np.array(bl).reshape(batch_size, sample_size, ft_size)

        if sparse:
            ba = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(ba))
            bd = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(bd))
        else:
            ba = torch.FloatTensor(ba)
            bd = torch.FloatTensor(bd)

        bf = torch.FloatTensor(bf)
        bl = torch.FloatTensor(bl)
        idx = np.random.permutation(sample_size)
        shuf_fts = bf[:, idx, :]
        shuf_fls = bl[:, idx, :]

        if cuda:
            bf = bf.cuda()
            bl = bl.cuda()
            ba = ba.cuda()
            bd = bd.cuda()
            shuf_fts = shuf_fts.cuda()
            shuf_fls = shuf_fls.cuda()

        model.train()
        optimiser.zero_grad()

        logits, logits1, logits2 = model(bf, bl, shuf_fts, shuf_fls, ba, bd, small_ratio, sparse)

        loss = b_xent(logits, lbl) + gamma * b_xent(logits1, lbl1) + sigma * b_xent(logits2, lbl2)

        loss.backward()
        optimiser.step()

        if verbose:
            print('Epoch: {0}, Loss: {1:0.4f}'.format(epoch, loss.item()))

        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), 'model.pkl')
        else:
            cnt_wait += 1

    #################################################################
    if verbose:
        print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load('model.pkl'))

    if sparse:
        adj = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(adj))
        diff = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(diff))

    features = torch.FloatTensor(features[np.newaxis])
    feat_L = torch.FloatTensor(feat_L[np.newaxis])
    adj = torch.FloatTensor(adj[np.newaxis])
    diff = torch.FloatTensor(diff[np.newaxis])
    if cuda:
        features = features.cuda()
        feat_L = feat_L.cuda()
        adj = adj.cuda()
        diff = diff.cuda()

    embeds = model.embed(features, feat_L, adj, diff, sparse)
    train_embs = embeds[0, idx_train]
    test_embs = embeds[0, idx_test]

    train_lbls = labels[idx_train]
    test_lbls = labels[idx_test]

    accs = []
    wd = 0.01 if dataset == 'citeseer' else 0.0

    for _ in range(50):
        log = LogReg(hid_units, nb_classes)
        opt = torch.optim.Adam(log.parameters(), lr=1e-2, weight_decay=wd)
        if cuda:
            log.cuda()
        for _ in range(300):
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        accs.append(acc * 100)

    accs = torch.stack(accs)
    print('ACC {}:'.format(accs.mean().item()))
    return None
