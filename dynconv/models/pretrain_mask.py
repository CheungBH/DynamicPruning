import torch
import torch.nn as nn


class ZeroRatioMask:
    def __init__(self, mask_thresh, target_stage, **kwargs):
        self.threshold = mask_thresh
        self.target_stage = target_stage

    def process(self, feat, stride, curr_block):
        bs, c, h, w = feat.shape
        zero_mask = torch.sum((feat == 0).int(), dim=1)
        soft_mask = torch.true_divide(zero_mask, c)
        if stride:
            soft_mask = nn.functional.upsample_nearest(soft_mask.unsqueeze(dim=1), size=(int(h/2), int(w/2)))
        if curr_block not in self.target_stage:
            return torch.ones_like(soft_mask)
        else:
            return (soft_mask <= self.threshold).int()


class ZeroRatioTopMask:
    def __init__(self, mask_thresh, target_stage, **kwargs):
        self.threshold = mask_thresh
        self.target_stage = target_stage

    def process(self, feat, stride, curr_block):
        bs, c, h, w = feat.shape
        zero_mask = torch.sum((feat == 0).int(), dim=1)
        soft_mask = torch.true_divide(zero_mask, c)
        if stride:
            soft_mask = nn.functional.upsample_nearest(soft_mask.unsqueeze(dim=1), size=(int(h/2), int(w/2))).squeeze(dim=1)
        if curr_block not in self.target_stage:
            return torch.ones_like(soft_mask)
        else:
            thresh = soft_mask.view(-1).sort()[0][int(soft_mask.numel()*self.threshold)]
            return (soft_mask <= thresh).int()


class SumNormalizeMask:
    def __init__(self, mask_thresh, target_stage, **kwargs):
        self.threshold = mask_thresh
        self.target_stage = target_stage

    def process(self, feat, stride, curr_block):
        summed_feat = torch.sum(feat, 1)
        if stride:
            b, c, h, w = feat.shape
            summed_feat = nn.functional.upsample_nearest(summed_feat, size=(int(h/2), int(w/2)))
        thresholds = summed_feat.view(feat.shape[0], -1).sort(dim=1)[0][:, int((summed_feat.shape[-1]*summed_feat.shape[-2]*self.threshold))]
        thresh_masks = thresholds.unsqueeze(dim=1).unsqueeze(dim=1).expand_as(summed_feat)

        if curr_block not in self.target_stage:
            return torch.ones_like(mask)
        else:
            return (summed_feat > thresh_masks).int()


class AbsSumNormalizeMask:
    def __init__(self, mask_thresh, target_stage, **kwargs):
        self.threshold = mask_thresh
        self.target_stage = target_stage

    def process(self, feat, stride, curr_block):
        summed_feat = torch.sum(torch.abs(feat), 1)
        if stride:
            b, c, h, w = feat.shape
            summed_feat = nn.functional.upsample_nearest(summed_feat, size=(int(h/2), int(w/2)))
        thresholds = summed_feat.view(feat.shape[0], -1).sort(dim=1)[0][:, int((summed_feat.shape[-1]*summed_feat.shape[-2]*self.threshold))]
        thresh_masks = thresholds.unsqueeze(dim=1).unsqueeze(dim=1).expand_as(summed_feat)

        if curr_block not in self.target_stage:
            return torch.ones_like(mask)
        else:
            return (summed_feat > thresh_masks).int()


class NoneMask:
    def __init__(self):
        pass

    def process(self, feat, stride, curr_block):
        bs, c, h, w = feat.shape
        if stride:
            return torch.ones((bs, 1, int(h/2), int(w/2))).cuda()
        return torch.ones(bs, 1, h, w).cuda()


class RandomMask:
    def __init__(self, mask_thresh, target_stage, **kwargs):
        self.threshold = mask_thresh
        self.target_stage = target_stage

    def process(self, feat, stride, curr_block):
        bs, c, h, w = feat.shape
        rand_mask = torch.rand((bs, 1, h, w)).cuda()
        if stride:
            rand_mask = nn.functional.upsample_nearest(rand_mask, size=(int(h/2), int(w/2)))
        if curr_block not in self.target_stage:
            return torch.ones_like(rand_mask)
        else:
            return (rand_mask > self.threshold).int()
