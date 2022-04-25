import numpy as np
import numbers

import torch
import torch.nn.functional as F
from torch import nn


class SemanticBatchHardXBM(nn.Module):
    def __init__(self, opt):
        super(SemanticBatchHardXBM, self).__init__()
        self.opt = opt
        self.fc = nn.Linear(in_features=opt.wv_size, out_features=self.opt.num_hashing)
        # self.bn = nn.BatchNorm1d(num_features=opt.num_hashing)

        # self.fc_2 = nn.Linear(in_features=opt.wv_size + opt.backbone_nhash, out_features=self.opt.backbone_nhash)
        # self.fc_back = nn.Linear(in_features=opt.backbone_nhash, out_features=self.opt.wv_size)
        # self.fc_consistency = nn.Linear(in_features=self.opt.backbone_nhash, out_features=self.opt.backbone_nhash, bias=False)
        # self.bn = MSSBN1d(num_features=self.opt.backbone_nhash)
        self.xbm = None
        if opt.K != 0:
            self.xbm = XBM(K=opt.K, feat_dim=opt.num_hashing)
        self.xbm_start_iter = opt.xbm_start_iter

    def forward(self, x, wv, label, iter=0):
        # x size: (Batch_size, Dim)
        # batch_size = x.shape[0]

        sem_org = self.fc(wv)
        x_prev, label_prev = x, label
        # sem_org = self.bn(sem_org)


        if self.xbm is not None:
            self.xbm.enqueue_dequeue(x_prev.detach(), sem_org.detach(), label_prev.detach())

        if self.xbm is None:
            alpha = torch.rand(x.size(0), 1).to(sem_org.get_device())
            sem = alpha * sem_org + (1.0 - alpha) * x

        else:
            if iter < self.xbm_start_iter:
                alpha = torch.rand(x.size(0), 1).to(sem_org.get_device())
                sem = alpha * sem_org + (1.0 - alpha) * x
            else:
                x_xbm, sem_org_xbm, label_xbm = self.xbm.get()
                x = torch.cat((x, x_xbm), 0)
                sem_org = torch.cat((sem_org, sem_org_xbm), 0)
                label = torch.cat((label, label_xbm), 0)
                alpha = torch.rand(x.size(0), 1).to(sem_org.get_device())
                sem = alpha * sem_org + (1.0 - alpha) * x
        dists = self._pairwise_distance(sem.cpu(), x.cpu(), 'euclidean').cuda()
        same_identity_mask = torch.eq(label.unsqueeze(dim=1), label.unsqueeze(dim=0))
        positive_mask = torch.logical_xor(same_identity_mask[:, -x.shape[0]:],
                                          torch.eye(label.size(0), dtype=torch.bool).to(label.get_device()))

        furthest_positive, _ = torch.max(dists * (positive_mask.float()), dim=1)
        closest_negative, _ = torch.min(dists + 1e8 * (same_identity_mask.float()), dim=1)


        diff = furthest_positive - closest_negative
        # if isinstance(self.opt.margin, numbers.Real):
        #     diff = F.relu(diff + self.opt.margin)
        # elif self.opt.margin == 'soft':
        diff = F.softplus(diff)

        return diff, sem

    def _pairwise_distance(self, x, sem, metric):
        diffs = x.unsqueeze(dim=1) - sem.unsqueeze(dim=0)
        if metric == 'sqeuclidean':
            return (diffs **2).sum(dim=-1)
        elif metric == 'euclidean':
            return torch.sqrt(((diffs **2) + 1e-16).sum(dim=-1))
        elif metric == 'cityblock':
            return diffs.abs().sum(dim=-1)

class XBM:
    def __init__(self, K, feat_dim):
        self.K = K
        self.feats = torch.zeros(self.K, feat_dim).cuda()
        self.sems = torch.zeros(self.K, feat_dim).cuda()
        self.targets = -1 * torch.ones(self.K, dtype=torch.long).cuda()
        self.ptr = 0

    @property
    def is_full(self):
        return self.targets[-1].item() != -1

    def get(self):
        if self.is_full:
            return self.feats, self.sems, self.targets
        else:
            return self.feats[:self.ptr], self.sems[:self.ptr], self.targets[:self.ptr]

    def enqueue_dequeue(self, feats, sems, targets):
        q_size = len(targets)
        if self.ptr + q_size > self.K:
            self.feats[-q_size:] = feats
            self.sems[-q_size:] = sems
            self.targets[-q_size:] = targets
            self.ptr = 0
        else:
            self.feats[self.ptr: self.ptr + q_size] = feats
            self.sems[self.ptr: self.ptr + q_size] = sems
            self.targets[self.ptr: self.ptr + q_size] = targets
            self.ptr += q_size



class SemanticBatchHardXBM2(nn.Module):
    def __init__(self, opt):
        super(SemanticBatchHardXBM2, self).__init__()
        self.opt = opt
        self.fc = nn.Linear(in_features=opt.wv_size, out_features=self.opt.num_hashing)
        # self.bn = nn.BatchNorm1d(num_features=opt.num_hashing)

        # self.fc_2 = nn.Linear(in_features=opt.wv_size + opt.backbone_nhash, out_features=self.opt.backbone_nhash)
        # self.fc_back = nn.Linear(in_features=opt.backbone_nhash, out_features=self.opt.wv_size)
        # self.fc_consistency = nn.Linear(in_features=self.opt.backbone_nhash, out_features=self.opt.backbone_nhash, bias=False)
        # self.bn = MSSBN1d(num_features=self.opt.backbone_nhash)
        self.xbm = XBM2(K=opt.K, feat_dim=opt.num_hashing)

    def forward(self, x, wv, label, iter=0):
        # x size: (Batch_size, Dim)
        # batch_size = x.shape[0]

        sem_org = self.fc(wv)
        x_prev, label_prev = x, label
        # sem_org = self.bn(sem_org)

        alpha = torch.rand(x.size(0), 1).to(sem_org.get_device())
        sem = alpha * sem_org + (1.0 - alpha) * x
        sem_prev = sem
        dists = self._pairwise_distance(sem, x, 'euclidean')
        same_identity_mask = torch.eq(label.unsqueeze(dim=1), label.unsqueeze(dim=0))
        positive_mask = torch.logical_xor(same_identity_mask[:, -x.shape[0]:],
                                          torch.eye(label.size(0), dtype=torch.bool).to(label.get_device()))

        furthest_positive, _ = torch.max(dists * (positive_mask.float()), dim=1)
        closest_negative, _ = torch.min(dists + 1e8 * (same_identity_mask.float()), dim=1)

        if iter > 1000:
            x_xbm, sem_xbm, label_xbm = self.xbm.get()
            dists2 = self._pairwise_distance(sem, x_xbm, 'euclidean')
            positive_mask2 = torch.eq(label.unsqueeze(dim=1), label_xbm.unsqueeze(dim=0))

            furthest_positive2, _ = torch.max(dists2 * (positive_mask2.float()), dim=1)
            closest_negative2, _ = torch.min(dists2 + 1e8 * (positive_mask2.float()), dim=1)
            furthest_positive = torch.max(furthest_positive, furthest_positive2)
            closest_negative = torch.min(closest_negative, closest_negative2)


        diff = furthest_positive - closest_negative
        # if isinstance(self.opt.margin, numbers.Real):
        #     diff = F.relu(diff + self.opt.margin)
        # elif self.opt.margin == 'soft':
        diff = F.softplus(diff)

        self.xbm.enqueue_dequeue(x_prev.detach(), sem_prev.detach(), label_prev.detach())
        return diff, sem

    def _pairwise_distance(self, x, sem, metric):
        diffs = x.unsqueeze(dim=1) - sem.unsqueeze(dim=0)
        if metric == 'sqeuclidean':
            return (diffs **2).sum(dim=-1)
        elif metric == 'euclidean':
            return torch.sqrt(((diffs **2) + 1e-16).sum(dim=-1))
        elif metric == 'cityblock':
            return diffs.abs().sum(dim=-1)

class XBM2:
    def __init__(self, K, feat_dim):
        self.K = K
        self.feats = torch.zeros(self.K, feat_dim).cuda()
        self.sems = torch.zeros(self.K, feat_dim).cuda()
        self.targets = -1 * torch.ones(self.K, dtype=torch.long).cuda()
        self.ptr = 0

    @property
    def is_full(self):
        return self.targets[-1].item() != -1

    def get(self):
        if self.is_full:
            return self.feats, self.sems, self.targets
        else:
            return self.feats[:self.ptr], self.sems[:self.ptr], self.targets[:self.ptr]

    def enqueue_dequeue(self, feats, sems, targets):
        q_size = len(targets)
        if self.ptr + q_size > self.K:
            self.feats[-q_size:] = feats
            self.sems[-q_size:] = sems
            self.targets[-q_size:] = targets
            self.ptr = 0
        else:
            self.feats[self.ptr: self.ptr + q_size] = feats
            self.sems[self.ptr: self.ptr + q_size] = sems
            self.targets[self.ptr: self.ptr + q_size] = targets
            self.ptr += q_size



