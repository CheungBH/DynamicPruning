import torch.nn as nn
import math
import torch


class ChannelVectorUnit(nn.Module):
    def __init__(self, in_channels, out_channels, group_size=1, pooling_method="ave", budget=1.0, **kwargs):
        super(ChannelVectorUnit, self).__init__()
        self.pooling = nn.AdaptiveMaxPool2d(1) if pooling_method == "max" else nn.AdaptiveAvgPool2d(1)
        self.group_size = group_size
        assert out_channels % group_size == 0, "The channels are not grouped with the same size"
        self.sigmoid = nn.Sigmoid()
        self.channel_saliency_predictor = nn.Linear(in_channels, out_channels//group_size)
        nn.init.kaiming_normal_(self.channel_saliency_predictor.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.channel_saliency_predictor.bias, 1.)
        self.sparsity = budget

    def forward(self, x, meta):
        x = self.pooling(x).squeeze()
        x = self.channel_saliency_predictor(x)
        x = self.sigmoid(x)
        meta["lasso_sum"] += torch.mean(torch.sum(x, dim=-1))
        x = self.winner_take_all(x)
        x = self.expand(x)
        return x

    def expand(self, x):
        bs, vec_size = x.shape
        return x.unsqueeze(dim=-1).expand(bs, vec_size, self.group_size).reshape(bs, vec_size*self.group_size)

    def winner_take_all(self, x):
        if self.sparsity >= 1.0:
            return x
        else:
            k = math.ceil((1 - self.sparsity) * x.size(-1))
            inactive_idx = (-x).topk(k - 1, 1)[1]
            zero_filtered = x.scatter_(1, inactive_idx, 0)
            return (zero_filtered > 0).int()


# def winner_take_all(x, sparsity_ratio):
#     if sparsity_ratio < 1.0:
#         k = math.ceil((1-sparsity_ratio) * x.size(-1))
#         inactive_idx = (-x).topk(k-1, 1)[1]
#         zero_filtered = x.scatter_(1, inactive_idx, 0)
#         return (zero_filtered > 0).int()
#     else:
#         return x


def conv1x1(conv_module, x, mask, fast=False):
    w = conv_module.weight.data
    mask.flops_per_position += w.shape[0]*w.shape[1]
    conv_module.__mask__ = mask
    return conv_module(x)


def conv3x3_dw(conv_module, x, mask_dilate, mask, fast=False):
    w = conv_module.weight.data
    mask.flops_per_position += w.shape[0]*w.shape[1]*w.shape[2]*w.shape[3]
    conv_module.__mask__ = mask
    return conv_module(x)


def conv3x3(conv_module, x, mask_dilate, mask, fast=False):
    w = conv_module.weight.data
    mask.flops_per_position += w.shape[0]*w.shape[1]*w.shape[2]*w.shape[3]
    conv_module.__mask__ = mask
    return conv_module(x)


## BATCHNORM and RELU
def bn_relu(bn_module, relu_module, x, mask, fast=False):
    bn_module.__mask__ = mask
    if relu_module is not None:
        relu_module.__mask__ = mask

    x = bn_module(x)
    x = relu_module(x) if relu_module is not None else x
    return x

def apply_saliency(x, vector):
    pass