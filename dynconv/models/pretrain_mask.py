import torch
import torch.nn as nn


class ZeroRatioMask:
    def __init__(self, mask_thresh, min_stage, **kwargs):
        self.threshold = mask_thresh
        self.min_stage = min_stage

    def process(self, feat, stride, curr_block):
        bs, c, h, w = feat.shape
        zero_mask = torch.sum((feat == 0).int(), dim=1)
        soft_mask = torch.true_divide(zero_mask, c)
        if stride:
            soft_mask = nn.functional.upsample_nearest(soft_mask.unsqueeze(dim=1), size=(int(h/2), int(w/2)))
        if curr_block < self.min_stage:
            return torch.ones_like(soft_mask)
        else:
            return (soft_mask <= self.threshold).int()


class ZeroRatioTopMask:
    def __init__(self, mask_thresh, min_stage, **kwargs):
        self.threshold = mask_thresh
        self.min_stage = min_stage

    def process(self, feat, stride, curr_block):
        bs, c, h, w = feat.shape
        zero_mask = torch.sum((feat == 0).int(), dim=1)
        soft_mask = torch.true_divide(zero_mask, c)
        if stride:
            soft_mask = nn.functional.upsample_nearest(soft_mask.unsqueeze(dim=1), size=(int(h/2), int(w/2)))
        if curr_block < self.min_stage:
            return torch.ones_like(soft_mask)
        else:
            thresh = soft_mask.view(-1).sort()[0][int(soft_mask.numel()*self.threshold)]
            return (soft_mask <= thresh).int()


class SumNormalizeMask:
    def __init__(self, mask_thresh, min_stage, **kwargs):
        self.threshold = mask_thresh
        self.min_stage = min_stage

    def process(self, feat, stride, curr_block):
        summed_feat = torch.sum(feat, 1).view(feat.shape[0], -1)
        min_value, max_value = torch.min(summed_feat, dim=1)[0], torch.max(summed_feat, dim=1)[0]
        normed_feat = ((summed_feat - min_value) / (max_value - min_value)).unsqueeze(dim=1)
        # normed_feat = nn.functional.normalize(torch.sum(feat, 1)).unsqueeze(dim=1)
        if stride:
            b, c, h, w = feat.shape
            normed_feat = nn.functional.upsample_nearest(normed_feat, size=(int(h/2), int(w/2)))
        if curr_block < self.min_stage:
            return torch.ones_like(normed_feat)
        else:
            return (normed_feat > self.threshold).int()


class NoneMask:
    def __init__(self):
        pass

    def process(self, feat, stride, min_stage):
        bs, c, h, w = feat.shape
        if stride:
            return torch.ones((bs, 1, int(h/2), int(w/2))).cuda()
        return torch.ones(bs, 1, h, w).cuda()

