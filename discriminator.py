import torch
import torch.nn as nn
from readout import Readout


def random_drop_feature(pos1, neg1, drop_prob=0.3):
    drop_mask = torch.empty(
        (pos1.size(1), pos1.size(2)),
        dtype=torch.float32).uniform_(0, 1) < drop_prob
    pos1 = pos1.clone()
    neg1 = neg1.clone()
    pos1[:, drop_mask] = 0
    neg1[:, drop_mask] = 0
    return pos1, neg1


class Exchange(nn.Module):
    def __init__(self):
        super(Exchange, self).__init__()

    def forward(self, x1, x2):
        return (x1 + x2)/2


class Select(nn.Module):
    def __init__(self):
        super(Select, self).__init__()

    def forward(self, X, idx):
        list_X = []
        idx, _ = torch.sort(idx, dim=0, descending=False)
        for i in range(idx.size()[0]):
            list_X.append(X[i][idx[i], :])
        new_x = torch.stack(list_X)
        return new_x


class Discriminator(nn.Module):
    def __init__(self, n_h, drop_rate=0.1):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        self.f_k1 = nn.Bilinear(n_h, n_h, 1)
        self.f_k2 = nn.Bilinear(n_h, n_h, 1)
        self.read = Readout()
        self.sigm = nn.Sigmoid()
        self.exchange = Exchange()
        self.select = Select()
        self.alpha = nn.Parameter(nn.init.constant_(torch.zeros(1), 0.5), requires_grad=True)
        self.beta = nn.Parameter(nn.init.constant_(torch.zeros(1), 0.5), requires_grad=True)
        self.lamda = nn.Parameter(nn.init.constant_(torch.zeros(1), 0.5), requires_grad=True)
        self.drop_rate = drop_rate
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, adj, diff, sub_local_pos1, sub_local_pos2, sub_local_neg1, sub_local_neg2, k):

        sub_local_pos1, sub_local_neg1 = random_drop_feature(sub_local_pos1, sub_local_neg1, drop_prob=self.drop_rate)
        sub_local_pos2, sub_local_neg2 = random_drop_feature(sub_local_pos2, sub_local_neg2, drop_prob=self.drop_rate)

        sub_fusion_pos = self.exchange(sub_local_pos1, sub_local_pos2)
        sub_fusion_neg = self.exchange(sub_local_neg1, sub_local_neg2)

        sub_global1 = self.sigm(self.read(sub_local_pos1))
        sub_global1 = torch.unsqueeze(sub_global1, 1).expand_as(sub_local_pos1).contiguous()
        sub_global2 = self.sigm(self.read(sub_local_pos2))
        sub_global2 = torch.unsqueeze(sub_global2, 1).expand_as(sub_local_pos2).contiguous()

        sub_mutual_pos1 = torch.squeeze(self.f_k(sub_fusion_pos, sub_global1), 2)
        sub_mutual_neg1 = torch.squeeze(self.f_k(sub_fusion_neg, sub_global1), 2)
        sub_mutual_pos2 = torch.squeeze(self.f_k(sub_fusion_pos, sub_global2), 2)
        sub_mutual_neg2 = torch.squeeze(self.f_k(sub_fusion_neg, sub_global2), 2)

        logits_fusion1 = torch.cat((sub_mutual_pos1, sub_mutual_neg1), 1)
        logits_fusion2 = torch.cat((sub_mutual_pos2, sub_mutual_neg2), 1)

        logits_fusion = self.alpha * logits_fusion1 + (1 - self.alpha) * logits_fusion2

        ####################################################

        mutual_score_pos = logits_fusion[:, :logits_fusion.shape[1] // 2]
        mutual_score_neg = logits_fusion[:, logits_fusion.shape[1] // 2:]

        mutual_score = mutual_score_neg - mutual_score_pos

        _, idx_pos = torch.topk(self.sigm(mutual_score), logits_fusion.shape[1] // 2, largest=True)

        idx = idx_pos[:, : int(k * (logits_fusion.shape[1] // 2))]

        select_X_pos = self.select(sub_fusion_pos, idx)
        select_X_neg = self.select(sub_fusion_neg, idx)

        sub_local1 = self.sigm(torch.bmm(adj, sub_local_pos1))
        sub_local2 = self.sigm(torch.bmm(diff, sub_local_pos2))

        sub_local1 = self.select(sub_local1, idx)
        sub_local2 = self.select(sub_local2, idx)

        select_X_mutual_pos1 = torch.squeeze(self.f_k1(select_X_pos, sub_local1), 2)
        select_X_mutual_neg1 = torch.squeeze(self.f_k1(select_X_neg, sub_local1), 2)
        select_X_mutual_pos2 = torch.squeeze(self.f_k1(select_X_pos, sub_local2), 2)
        select_X_mutual_neg2 = torch.squeeze(self.f_k1(select_X_neg, sub_local2), 2)

        logits_fusion_sub1 = torch.cat((select_X_mutual_pos1, select_X_mutual_neg1), 1)
        logits_fusion_sub2 = torch.cat((select_X_mutual_pos2, select_X_mutual_neg2), 1)

        logits_fusion_sub = self.beta * logits_fusion_sub1 + (1 - self.beta) * logits_fusion_sub2

        ####################################################

        mutual_score_pos_sub = logits_fusion_sub[:, :logits_fusion_sub.shape[1] // 2]
        mutual_score_neg_sub = logits_fusion_sub[:, logits_fusion_sub.shape[1] // 2:]

        mutual_score_sub = mutual_score_neg_sub - mutual_score_pos_sub

        _, idx_pos_sub = torch.topk(self.sigm(mutual_score_sub), logits_fusion_sub.shape[1] // 2, largest=True)

        idx_sub = idx_pos_sub[:, : int(k * (logits_fusion_sub.shape[1] // 2))]

        select_X_pos_sub = self.select(select_X_pos, idx_sub)
        select_X_neg_sub = self.select(select_X_neg, idx_sub)

        sub_local1_sub = self.select(self.select(self.sigm(sub_local_pos1), idx), idx_sub)
        sub_local2_sub = self.select(self.select(self.sigm(sub_local_pos2), idx), idx_sub)

        select_X_mutual_pos_sub1 = torch.squeeze(self.f_k2(select_X_pos_sub, sub_local1_sub), 2)
        select_X_mutual_neg_sub1 = torch.squeeze(self.f_k2(select_X_neg_sub, sub_local1_sub), 2)
        select_X_mutual_pos_sub2 = torch.squeeze(self.f_k2(select_X_pos_sub, sub_local2_sub), 2)
        select_X_mutual_neg_sub2 = torch.squeeze(self.f_k2(select_X_neg_sub, sub_local2_sub), 2)

        logits_fusion_sub_sub1 = torch.cat((select_X_mutual_pos_sub1, select_X_mutual_neg_sub1), 1)
        logits_fusion_sub_sub2 = torch.cat((select_X_mutual_pos_sub2, select_X_mutual_neg_sub2), 1)
        logits_fusion_sub_sub = self.lamda * logits_fusion_sub_sub1 + (1 - self.lamda) * logits_fusion_sub_sub2

        return logits_fusion, logits_fusion_sub, logits_fusion_sub_sub
