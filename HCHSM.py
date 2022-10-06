import torch
import torch.nn as nn
from discriminator import Discriminator


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        return self.act(out)


class Model(nn.Module):
    def __init__(self, n_in, n_h, drop_rate=0.1):
        super(Model, self).__init__()
        self.gcn1 = GCN(n_in, n_h)
        self.gcn2 = GCN(n_in, n_h)
        self.disc = Discriminator(n_h, drop_rate)

    def forward(self, bf, bl, shuf_fts, shuf_fls, adj, diff, small_ratio, sparse):
        h_1 = self.gcn1(bf, adj, sparse)
        h_2 = self.gcn2(bl, diff, sparse)
        h_3 = self.gcn1(shuf_fts, adj, sparse)
        h_4 = self.gcn2(shuf_fls, diff, sparse)

        logits_fusion, logits_fusion_sub, logits_fusion_sub_sub = self.disc(adj, diff, h_1, h_2, h_3, h_4, small_ratio)

        return logits_fusion, logits_fusion_sub, logits_fusion_sub_sub

    def embed(self, bf, bl, adj, diff, sparse):
        h_1 = self.gcn1(bf, adj, sparse)
        h_2 = self.gcn2(bl, diff, sparse)
        return (h_1 + h_2).detach()


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)
        self.sigm = nn.Sigmoid()

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = torch.log_softmax(self.fc(seq), dim=-1)
        return ret
