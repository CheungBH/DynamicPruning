import torch.nn as nn
import math
import torch


class MaskedAvePooling(nn.Module):
    def __init__(self, size=1):
        super(MaskedAvePooling, self).__init__()
        self.pooling = nn.AdaptiveAvgPool2d(size)

    def forward(self, x, mask):
        pooled_feat = self.pooling(x * mask.expand_as(x))
        total_pixel_num = mask.shape[-1] * mask.shape[-2]
        active_pixel_num = mask.view(x.shape[0], -1).sum(dim=1)
        active_mask = active_pixel_num.unsqueeze(dim=1).unsqueeze(dim=1).unsqueeze(dim=1).expand_as(pooled_feat)
        return (pooled_feat * total_pixel_num)/active_mask


class ChannelVectorUnit(nn.Module):
    def __init__(self, in_channels, out_channels, group_size=1, pooling_method="ave", channel_budget=1.0, channel_stage=[-1],
                 **kwargs):
        super(ChannelVectorUnit, self).__init__()
        self.pooling = nn.AdaptiveMaxPool2d(1) if pooling_method == "max" else MaskedAvePooling()
        self.out_channels = out_channels
        self.group_size = group_size
        assert out_channels % group_size == 0, "The channels are not grouped with the same size"
        self.sigmoid = nn.Sigmoid()
        self.channel_saliency_predictor = nn.Linear(in_channels, out_channels//group_size)
        nn.init.kaiming_normal_(self.channel_saliency_predictor.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.channel_saliency_predictor.bias, 1.)
        self.sparsity = channel_budget
        self.target_stage = channel_stage

    def forward(self, x, meta):
        if meta["stage_id"] not in self.target_stage:
            return torch.ones(x.shape[0], self.out_channels).cuda()
        if isinstance(self.pooling, MaskedAvePooling):
            x = self.pooling(x, meta["masked_feat"]).squeeze()
        else:
            x = self.pooling(x).squeeze()
        x = self.channel_saliency_predictor(x)
        x = self.sigmoid(x)
        meta["lasso_sum"] += torch.mean(torch.sum(x, dim=-1))
        x = self.winner_take_all(x.clone())
        x = self.expand(x)
        return x

    def expand(self, x):
        bs, vec_size = x.shape
        return x.unsqueeze(dim=-1).expand(bs, vec_size, self.group_size).reshape(bs, vec_size*self.group_size)

    def winner_take_all(self, x):
        if self.sparsity >= 1.0:
            return x
        elif x.size(-1) == 1:
            return (x > -1).int()
        else:
            k = math.ceil((1 - self.sparsity) * x.size(-1))
            inactive_idx = (-x).topk(k)[1]
            zero_filtered = x.scatter_(1, inactive_idx, 0)
            return (zero_filtered > 0).int()


def conv_forward(conv_module, x, inp_vec=None, out_vec=None):
    conv_module.__input_ratio__ = vector_ratio(inp_vec)
    conv_module.__output_ratio__ = vector_ratio(out_vec)
    return conv_module(x)


def bn_relu_foward(bn_module, relu_module, x, vector=None):
    bn_module.__output_ratio__ = vector_ratio(vector)
    if relu_module is not None:
        relu_module.vector = vector

    x = bn_module(x)
    x = relu_module(x) if relu_module is not None else x
    return x


def channel_process(x, vector):
    if len(vector.shape) != 2:
        return x * vector
    else:
        return x * vector.unsqueeze(-1).unsqueeze(-1).expand_as(x)


def vector_ratio(vector):
    if vector is None:
        return 1
    return torch.true_divide(vector.sum(), vector.numel()).tolist()

# def winner_take_all(x, sparsity_ratio):
#     if sparsity_ratio < 1.0:
#         k = math.ceil((1-sparsity_ratio) * x.size(-1))
#         inactive_idx = (-x).topk(k-1, 1)[1]
#         zero_filtered = x.scatter_(1, inactive_idx, 0)
#         return (zero_filtered > 0).int()
#     else:
#         return x
