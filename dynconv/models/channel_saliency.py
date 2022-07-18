import torch.nn as nn
import math
import torch
from torch.autograd import Variable


class GumbelSoftmax(nn.Module):
    '''
        gumbel softmax gate.
    '''

    def __init__(self, eps=1.0):
        super(GumbelSoftmax, self).__init__()
        self.eps = eps
        self.sigmoid = nn.Sigmoid()

    def gumbel_sample(self, template_tensor, eps=1e-8):
        uniform_samples_tensor = template_tensor.clone().uniform_()
        gumble_samples_tensor = torch.log(uniform_samples_tensor + eps) - torch.log(
            1 - uniform_samples_tensor + eps)
        return gumble_samples_tensor

    def gumbel_softmax(self, logits):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        gsamples = self.gumbel_sample(logits.data)
        logits = logits + Variable(gsamples)
        soft_samples = self.sigmoid(logits / self.eps)
        return soft_samples, logits

    def forward(self, logits):
        if not self.training:
            out_hard = (logits >= 0).float()
            return out_hard
        out_soft, prob_soft = self.gumbel_softmax(logits)
        out_hard = ((out_soft >= 0.5).float() - out_soft).detach() + out_soft
        return out_hard


def expand(x, group_size):
    bs, vec_size = x.shape
    return x.unsqueeze(dim=-1).expand(bs, vec_size, group_size).reshape(bs, vec_size*group_size)


class GumbelChannelUnit(nn.Module):
    '''
        Attention Mask.
    '''

    def __init__(self, inplanes, outplanes, group_size=4, eps=0.66667, budget=-1, bias=-1, channel_stage=[-1], **kwargs):
        super(GumbelChannelUnit, self).__init__()
        # Parameter
        # self.bottleneck = inplanes // fc_reduction
        self.inplanes, self.outplanes = inplanes, outplanes
        self.eleNum_c = torch.Tensor([outplanes])
        # channel attention
        self.pooling = MaskedAvePooling()
        self.channel_saliency_predictor = nn.Linear(inplanes, outplanes//group_size)
        self.target_stage = channel_stage
        self.group_size = group_size

        # if bias >= 0:
        #     nn.init.constant_(self.atten_c[3].bias, bias)
        # Gate
        self.gumbel = GumbelSoftmax(eps=eps)
        # Norm
        # self.norm = lambda x: torch.norm(x, p=1, dim=(1, 2, 3))

    def forward(self, x, meta):
        if meta["stage_id"] not in self.target_stage:
            return torch.ones(x.shape[0], self.outplanes).cuda(), meta
        batch, channel, _, _ = x.size()
        context = self.pooling(meta["saliency_mask"], meta["masks"][-1]["std"]).view(batch, -1)
        # transform
        # context = context.unsqueeze(dim=0) if batch == 1 else context
        c_in = self.channel_saliency_predictor(context)  # [N, C_out, 1, 1]
        # channel gate
        mask_c = self.gumbel(c_in)  # [N, C_out, 1, 1]
        meta["channel_prediction"][(meta["stage_id"], meta["block_id"])] = mask_c
        mask_c = expand(mask_c, self.group_size)
        return mask_c, meta


class MaskedAvePooling(nn.Module):
    def __init__(self, size=1):
        super(MaskedAvePooling, self).__init__()
        self.pooling = nn.AdaptiveAvgPool2d(size)

    def forward(self, x, mask):
        if mask is None:
            return self.pooling(x)
        mask = mask.hard
        pooled_feat = self.pooling(x * mask.expand_as(x))
        total_pixel_num = mask.shape[-1] * mask.shape[-2]
        active_pixel_num = mask.view(x.shape[0], -1).sum(dim=1)
        active_mask = active_pixel_num.unsqueeze(dim=1).unsqueeze(dim=1).unsqueeze(dim=1).expand_as(pooled_feat) + 1e-4
        return (pooled_feat * total_pixel_num)/active_mask


class MaskedMaxPooling(nn.Module):
    def __init__(self, size=1):
        super(MaskedMaxPooling, self).__init__()
        self.pooling = nn.AdaptiveAvgPool2d(size)

    def forward(self, feat, mask):
        if mask is None:
            return self.pooling(feat)
        else:
            return self.pooling(feat * mask.expand_as(feat))


class ChannelVectorUnit(nn.Module):
    def __init__(self, in_channels, out_channels, group_size=1, pooling_method="ave", channel_budget=1.0,
                 channel_stage=[-1], **kwargs):
        super(ChannelVectorUnit, self).__init__()
        self.pooling = MaskedMaxPooling() if pooling_method == "max" else MaskedAvePooling()
        self.out_channels = out_channels
        self.group_size = group_size
        # assert out_channels % group_size == 0, "The channels are not grouped with the same size"
        self.sigmoid = nn.Sigmoid()
        self.channel_saliency_predictor = nn.Linear(in_channels, out_channels//group_size)
        nn.init.kaiming_normal_(self.channel_saliency_predictor.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.channel_saliency_predictor.bias, 1.)
        self.sparsity = channel_budget
        self.target_stage = channel_stage

    def forward(self, x, meta):
        if meta["stage_id"] not in self.target_stage:
            return torch.ones(x.shape[0], self.out_channels).cuda(), meta
        x = self.pooling(meta["saliency_mask"], meta["masks"][-1]["std"]).squeeze()
        x = self.channel_saliency_predictor(x)
        x = self.sigmoid(x)
        meta["lasso_sum"] += torch.mean(torch.sum(x, dim=-1))
        x = x.unsqueeze(dim=0) if len(x.shape) == 1 else x
        x = self.winner_take_all(x.clone())
        meta["channel_prediction"][(meta["stage_id"], meta["block_id"])] = x
        x = self.expand(x)
        return x, meta

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


def conv_forward(conv_module, x, inp_vec=None, out_vec=None, forward=True):
    conv_module.__input_ratio__ = vector_ratio(inp_vec)
    conv_module.__output_ratio__ = vector_ratio(out_vec)
    if forward:
        return conv_module(x)


def bn_relu_foward(bn_module, relu_module, x, vector=None):
    bn_module.__output_ratio__ = vector_ratio(vector)
    relu_module.__output_ratio__ = vector_ratio(vector)
    if relu_module is not None:
        relu_module.vector = vector


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
