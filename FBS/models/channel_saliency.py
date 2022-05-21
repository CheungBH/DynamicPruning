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
