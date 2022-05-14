import torch.nn as nn


class ChannelVectorUnit(nn.Module):
    def __init__(self, in_channels, out_channels, group_size=1, pooling_method="ave", **kwargs):
        super(ChannelVectorUnit, self).__init__()
        self.pooling = nn.MaxPool2d(1) if pooling_method == "max" else nn.AdaptiveAvgPool2d(1)
        self.group_size = group_size
        self.sigmoid = nn.Sigmoid()
        self.channel_saliency_predictor = nn.Linear(in_channels, out_channels//group_size)
        nn.init.kaiming_normal_(self.channel_saliency_predictor.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.channel_saliency_predictor.bias, 1.)

    def forward(self, x, meta):
        x = self.pooling(x)
        x = self.channel_saliency_predictor(x)
        return self.sigmoid(x)


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