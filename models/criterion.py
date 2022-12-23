#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseLoss(nn.Module):
    def __init__(self):
        super(BaseLoss, self).__init__()

    def forward(self, preds, targets, weight=None):
        if isinstance(preds, list):
            N = len(preds)
            if weight is None:
                weight = preds[0].new_ones(1)

            errs = [self._forward(preds[n], targets[n], weight[n])
                    for n in range(N)]
            err = torch.mean(torch.stack(errs))

        elif isinstance(preds, torch.Tensor):
            if weight is None:
                weight = preds.new_ones(1)
            err = self._forward(preds, targets, weight)
        return err


class L1Loss(BaseLoss):
    def __init__(self):
        super(L1Loss, self).__init__()

    def _forward(self, pred, target, weight):
        return torch.mean(weight * torch.abs(pred - target))


class L2Loss(BaseLoss):
    def __init__(self):
        super(L2Loss, self).__init__()

    def _forward(self, pred, target, weight):
        return torch.mean(weight * torch.pow(pred - target, 2))


class SISNRLoss(BaseLoss):
    def __init__(self, eps=1e-6):
        super(SISNRLoss, self).__init__()
        self.eps = eps

    def forward(self, preds, targets):
        assert targets.size() == preds.size()  # (B, L)
        # Step 1. Zero-mean norm
        targets = targets - torch.mean(targets, axis=-1, keepdim=True)
        preds = preds - torch.mean(preds, axis=-1, keepdim=True)
        # Step 2. SI-SNR
        # s_target = <s', s>s / ||s||^2
        ref_energy = torch.sum(targets ** 2, axis=-1, keepdim=True) + self.eps
        proj = torch.sum(targets * preds, axis=-1, keepdim=True) * targets / ref_energy
        # e_noise = s' - s_target
        noise = preds - proj
        # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
        ratio = torch.sum(proj ** 2, axis=-1) / (torch.sum(noise ** 2, axis=-1) + self.eps)
        sisnr = 10 * torch.log10(ratio + self.eps)
        return -torch.mean(sisnr)


class CosineDistanceLoss(BaseLoss):
    def __init__(self):
        super(CosineDistanceLoss, self).__init__()

    def _forward(self, pred, target, weight):
        return - torch.mean(weight[:, 0, :, :] * (pred[:, 0, :, :] * target[:, 0, :, :] +
                                                  pred[:, 1, :, :] * target[:, 1, :, :]))


class BCELoss(BaseLoss):
    def __init__(self):
        super(BCELoss, self).__init__()

    def _forward(self, pred, target, weight):
        return F.binary_cross_entropy(pred, target, weight=weight)


class BCEWithLogitsLoss(BaseLoss):
    def __init__(self):
        super(BCEWithLogitsLoss, self).__init__()

    def _forward(self, pred, target, weight):
        return F.binary_cross_entropy_with_logits(pred, target, weight=weight)


class CELoss(BaseLoss):
    def __init__(self):
        super(CELoss, self).__init__()

    def _forward(self, pred, target, weight=None):
        return F.cross_entropy(pred, target)


class TripletLossCosine(BaseLoss):  # 要改一下!!!
    """
    Triplet loss with cosine distance
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    anchor: (B, 640, 1, 64), positive, negative: (B, 640, 1, 1), anchor最后一维每一个特征与pos,neg计算相似度，64个平均
    """

    def __init__(self, margin):
        super(TripletLossCosine, self).__init__()
        self.margin = margin

    def _forward(self, anchor, positive, negative, size_average=True):
        # anchor: (B, 640, 1, 64), positive: (B, 640, 1, 1)
        sim_pos = torch.mean(F.cosine_similarity(anchor, positive, dim=1).squeeze(1), dim=-1)  # (B,)
        sim_neg = torch.mean(F.cosine_similarity(anchor, negative, dim=1).squeeze(1), dim=-1)  # (B,)
        dis_pos = 1 - sim_pos  # (B,)
        dis_neg = 1 - sim_neg  # (B,)
        losses = F.relu((dis_pos - dis_neg) + self.margin)
        return losses.mean() if size_average else losses.sum()


class TripletLossCosine2(BaseLoss):
    """
    Triplet loss with cosine distance
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    anchor: (B, 640, 1, 64), positive, negative: (B, 640, 1, 1), anchor最后一维先平均, 再与pos,neg计算相似度
    """

    def __init__(self, margin):
        super(TripletLossCosine2, self).__init__()
        self.margin = margin

    def _forward(self, anchor, positive, negative, size_average=True):
        # anchor: (B, 640, 1, 64), positive: (B, 640, 1, 1), negative: (B, 640, 1, 1)
        anchor_ = torch.mean(anchor, dim=3)  # (B, 640, 1)
        pos_ = positive.squeeze(3)  # (B, 640, 1)
        neg_ = negative.squeeze(3)  # (B, 640, 1)
        sim_pos = F.cosine_similarity(anchor_, pos_, dim=1).squeeze(1)  # (B,)
        sim_neg = F.cosine_similarity(anchor_, neg_, dim=1).squeeze(1)  # (B,)
        dis_pos = 1 - sim_pos  # (B,)
        dis_neg = 1 - sim_neg  # (B,)
        losses = F.relu((dis_pos - dis_neg) + self.margin)
        return losses.mean() if size_average else losses.sum()


class TripletLoss(BaseLoss):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def _forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class NCELoss(BaseLoss):
    # visual: (B, 640, 1, 64), audio: (B, 640, 1, 1),  visual最后一维64个feature里的每一个与audio计算相似度, 64个相似度平均
    def __init__(self, temp):
        super(NCELoss, self).__init__()
        self.temp = temp

    def _dot_sim(self, feat1, feat2):
        # feat1: (B, 64, 640),  feat2: (B, 640, 1)
        sim = torch.mean(torch.matmul(feat1, feat2).squeeze(2) / self.temp, dim=1)  # (B,)
        return sim

    def forward(self, visual1, visual2, audio1, audio2):
        # visual: (B, 640, 1, 64),  audio: (B, 640, 1, 1)
        batch = len(visual1)
        visual1_ = F.normalize(visual1.squeeze(2).transpose(1, 2), p=2, dim=2)  # (B, 64, 640)
        visual2_ = F.normalize(visual2.squeeze(2).transpose(1, 2), p=2, dim=2)  # (B, 64, 640)
        audio1_ = F.normalize(audio1.squeeze(2), p=2, dim=1)  # (B, 640, 1)
        audio2_ = F.normalize(audio2.squeeze(2), p=2, dim=1)  # (B, 640, 1)
        pos_sim1 = self._dot_sim(visual1_, audio1_)  # (B, )
        pos_sim2 = self._dot_sim(visual2_, audio2_)  # (B, )
        neg_sim1 = self._dot_sim(visual1_, audio2_)  # (B, )
        neg_sim2 = self._dot_sim(visual2_, audio1_)  # (B, )
        pos_idx = torch.tensor([[1, 1, 0, 0]] * batch).int().cuda()
        sims = torch.stack((pos_sim1, pos_sim2, neg_sim1, neg_sim2), dim=1)
        softmax_sim = torch.softmax(sims, dim=1)
        loss = torch.sum(torch.mul(softmax_sim, pos_idx), dim=1)
        return loss.mean()


class NCELoss2(BaseLoss):
    # visual: (B, 640, 1, 64),  audio: (B, 640, 1, 1),  visual特征最后一维先平均,再与audio特征计算相似度
    def __init__(self, temp):
        super(NCELoss2, self).__init__()
        self.temp = temp

    def _dot_sim(self, feat1, feat2):
        # feat1: (B, 1, 640),  feat2: (B, 640, 1)
        sim = torch.matmul(feat1, feat2).squeeze(2).squeeze(1) / self.temp  # (B,)
        return sim

    def forward(self, visual1, visual2, audio1, audio2):
        # visual: (B, 640, 1, 64),  audio: (B, 640, 1, 1), audio_mix: (B, 640, 1, 1)
        batch = len(visual1)
        visual1_ = torch.mean(visual1, dim=3)  # (B, 640, 1)
        visual2_ = torch.mean(visual2, dim=3)  # (B, 640, 1)
        visual1_ = F.normalize(visual1_, p=2, dim=1).transpose(1, 2)  # (B, 1, 640)
        visual2_ = F.normalize(visual2_, p=2, dim=1).transpose(1, 2)  # (B, 1, 640)
        audio1_ = F.normalize(audio1.squeeze(3), p=2, dim=1)  # (B, 640, 1)
        audio2_ = F.normalize(audio2.squeeze(3), p=2, dim=1)  # (B, 640, 1)
        pos_sim1 = self._dot_sim(visual1_, audio1_)  # (B, )
        pos_sim2 = self._dot_sim(visual2_, audio2_)  # (B, )
        neg_sim1 = self._dot_sim(visual1_, audio2_)  # (B, )
        neg_sim2 = self._dot_sim(visual2_, audio1_)  # (B, )
        pos_idx = torch.tensor([[1, 1, 0, 0]] * batch).int().cuda()  # (B, 4)
        sims = torch.stack((pos_sim1, pos_sim2, neg_sim1, neg_sim2), dim=1)  # (B, 4)
        softmax_sim = torch.softmax(sims, dim=1)
        loss = torch.sum(torch.mul(softmax_sim, pos_idx), dim=1)
        return loss.mean()


class DistillLoss(BaseLoss):
    def __init__(self):
        super(DistillLoss, self).__init__()

    def _forward(self, pred, target, weight):
        pred = F.log_softmax(pred, dim=-1)
        target = F.softmax(target, dim=-1)
        return torch.mean(-torch.sum(torch.mul(pred, target), dim=-1)) * weight


class DistillKL(BaseLoss):
    def __init__(self):
        super(DistillKL, self).__init__()

    def _forward(self, pred, targ, weight):
        pred = F.log_softmax(pred, dim=-1)
        targ = F.softmax(targ, dim=-1)
        return F.kl_div(pred, targ, reduction='mean') * weight


class DistillKL2(BaseLoss):
    def __init__(self):
        super(DistillKL2, self).__init__()
        # 限定为输出logits使用

    def _forward(self, pred, targ, weight):  # pred: (B, L, C)
        shape = targ.shape
        targ = targ.reshape((shape[0] * shape[1], shape[2]))  # (B * L, C)
        idxes = torch.argmax(targ, dim=-1)  # (B * L)
        idxes = torch.where(idxes > 4, torch.tensor(1).cuda(), torch.tensor(0).cuda()).bool()

        pred = pred.reshape((shape[0] * shape[1], shape[2]))[idxes]  # (N, C)
        targ = targ[idxes]  # (N, C)
        pred = F.log_softmax(pred, dim=-1)
        targ = F.softmax(targ, dim=-1)
        return F.kl_div(pred, targ, reduction='mean') * weight
