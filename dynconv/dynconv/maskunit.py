import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import logger

dilate = False


class Mask():
    '''
    Class that holds the mask properties

    hard: the hard/binary mask (1 or 0), 4-dim tensor
    soft (optional): the float mask, same shape as hard
    active_positions: the amount of positions where hard == 1
    total_positions: the total amount of positions
                        (typically batch_size * output_width * output_height)
    '''

    def __init__(self, hard, soft=None):
        assert hard.dim() == 4
        assert hard.shape[1] == 1
        assert soft is None or soft.shape == hard.shape

        self.hard = hard
        self.active_positions = torch.sum(hard)  # this must be kept backpropagatable!
        self.total_positions = hard.numel()
        self.soft = soft

        self.flops_per_position = 0

    def size(self):
        return self.hard.shape

    def __repr__(self):
        return f'Mask with {self.active_positions}/{self.total_positions} positions, and {self.flops_per_position} accumulated FLOPS per position'


class StatMaskUnit(nn.Module):
    def __init__(self, init_thresh=0.5, stride=1, dilate_stride=1, **kwargs):
        super(StatMaskUnit, self).__init__()
        self.threshold = nn.Parameter(init_thresh * torch.ones(1, 1, 1, 1))
        self.threshold.requires_grad = False
        self.expandmask = ExpandMask(stride=dilate_stride)
        self.stride = stride

    def forward(self, x, meta):
        _, channel_num, orig_h, orig_w = x.shape
        summed_mask = torch.sum((x == 0).int(), dim=1)
        target_h, target_w = int(orig_h/self.stride), int(orig_w/self.stride)
        soft = torch.true_divide(summed_mask, channel_num).unsqueeze(dim=1)
        soft = nn.functional.upsample_nearest(soft, size=(target_h, target_w))

        hard = (soft > self.threshold).int()
        mask = Mask(hard, soft)

        hard_dilate = self.expandmask(mask.hard)
        mask_dilate = Mask(hard_dilate)

        if dilate:
            m = {'std': mask, 'dilate': mask_dilate}
        else:
            m = {'std': mask, 'dilate': mask}
        meta['masks'].append(m)
        return m


class StatMaskUnitMom(nn.Module):
    def __init__(self, budget=0.5, mask_thresh=0.5, stride=1, dilate_stride=1, momentum=0.9,
                 individual_forward=False, **kwargs):
        super(StatMaskUnitMom, self).__init__()
        self.threshold = nn.Parameter(mask_thresh * torch.ones(1, 1, 1, 1))
        self.expandmask = ExpandMask(stride=dilate_stride)
        self.stride = stride
        self.momentum = momentum
        self.budget = budget
        self.ind_for = individual_forward

    def sample_mask_forward(self, threshs, masks):
        _, c, w, h = masks.shape
        hard_masks = torch.zeros(1, c, w, h).cuda()
        for thresh, mask in zip(threshs, masks):
            hard_mask = (mask <= thresh).unsqueeze(dim=0)
            hard_masks = torch.cat((hard_masks, hard_mask), dim=0)
        return hard_masks[1:].int()

    def forward(self, x, meta):
        bs, channel_num, orig_h, orig_w = x.shape
        summed_mask = torch.sum((x == 0).int(), dim=1)
        target_h, target_w = int(orig_h / self.stride), int(orig_w / self.stride)
        target_index = int(target_h * target_w * self.budget)
        soft = torch.true_divide(summed_mask, channel_num).unsqueeze(dim=1)
        soft = nn.functional.upsample_nearest(soft, size=(target_h, target_w))
        if self.training:
            sorted_values, _ = torch.sort(soft.view(bs, -1), dim=1)
            sample_thresh = sorted_values[:, target_index]
            target_thresh = torch.mean(sample_thresh)
            if self.ind_for:
                hard = self.sample_mask_forward(sample_thresh, soft)
            else:
                hard = (soft <= target_thresh).int()
            updated_thresh = self.threshold * self.momentum + torch.ones(1).cuda() * target_thresh * (1 - self.momentum)
            self.threshold = nn.Parameter(updated_thresh.data * torch.ones(1, 1, 1, 1).cuda())
        else:
            hard = (soft <= self.threshold).int()

        mask = Mask(hard, soft)

        hard_dilate = self.expandmask(mask.hard)
        mask_dilate = Mask(hard_dilate)

        if dilate:
            m = {'std': mask, 'dilate': mask_dilate}
        else:
            m = {'std': mask, 'dilate': mask}
        meta['masks'].append(m)
        return m


class MaskUnit(nn.Module):
    '''
    Generates the mask and applies the gumbel softmax trick
    '''

    def __init__(self, channels, stride=1, dilate_stride=1, no_attention=False, mask_kernel=3, random_mask_stage=[-1],
                 budget=0.5, skip_layer_thresh=-1, **kwargs):
        super(MaskUnit, self).__init__()
        self.maskconv = Squeeze(channels=channels, stride=stride, mask_kernel=mask_kernel, no_attention=no_attention)
        self.gumbel = Gumbel()
        self.stride = stride
        self.random_mask_stage = random_mask_stage
        if dilate:
            self.expandmask = ExpandMask(stride=dilate_stride)
        self.budget = budget
        self.skip_layer_thresh = skip_layer_thresh + 1e-8

    def forward(self, x, meta):
        bs, _, w, h = x.shape
        if meta["stage_id"] in self.random_mask_stage and not self.training:
            soft = torch.rand(bs, 1, int(w/self.stride), int(h/self.stride)).cuda() - (1 - self.budget)
        else:
            soft = self.maskconv(x)
        hard = self.gumbel(soft, meta['gumbel_temp'], meta['gumbel_noise'])
        hard = self.skip_whole(hard)
        mask = Mask(hard, soft)

        if dilate:
            hard_dilate = self.expandmask(mask.hard)
            mask_dilate = Mask(hard_dilate)
            m = {'std': mask, 'dilate': mask_dilate}
        else:
            m = {'std': mask, 'dilate': mask}
        meta['masks'].append(m)
        return m

    def skip_whole(self, mask):
        if self.skip_layer_thresh <= 0:
            return mask
        else:
            percent = torch.true_divide(mask.sum(), mask.numel())
            return 0.5 * torch.sign(mask - 0.5*(torch.sign(self.skip_layer_thresh - percent)+1) - 1e-8) + 0.5

## Gumbel

class Gumbel(nn.Module):
    '''
    Returns differentiable discrete outputs. Applies a Gumbel-Softmax trick on every element of x.
    '''

    def __init__(self, eps=1e-8):
        super(Gumbel, self).__init__()
        self.eps = eps

    def forward(self, x, gumbel_temp=1.0, gumbel_noise=True):
        if not self.training:  # no Gumbel noise during inference
            return (x >= 0).float()

        logger.add('gumbel_noise', gumbel_noise)
        logger.add('gumbel_temp', gumbel_temp)

        if gumbel_noise:
            eps = self.eps
            U1, U2 = torch.rand_like(x), torch.rand_like(x)
            g1, g2 = -torch.log(-torch.log(U1 + eps) + eps), - \
                torch.log(-torch.log(U2 + eps) + eps)
            x = x + g1 - g2

        soft = torch.sigmoid(x / gumbel_temp)
        hard = ((soft >= 0.5).float() - soft).detach() + soft
        assert not torch.any(torch.isnan(hard))
        return hard


## Mask convs
class Squeeze(nn.Module):
    """
    Squeeze module to predict masks
    """

    def __init__(self, channels, stride=1, mask_kernel=3, no_attention=False):
        padding = 1 if mask_kernel == 3 else 0
        super(Squeeze, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.no_attention = no_attention
        if not self.no_attention:
            self.fc = nn.Linear(channels, 1, bias=True)
        self.conv = nn.Conv2d(channels, 1, stride=stride,
                              kernel_size=mask_kernel, padding=padding, bias=True)

    def forward(self, x):
        b, c, _, _ = x.size()
        if self.no_attention:
            return self.conv(x)
        else:
            y = self.avg_pool(x).view(b, c)
            y = self.fc(y).view(b, 1, 1, 1)
            z = self.conv(x)
            return z + y.expand_as(z)


class ExpandMask(nn.Module):
    def __init__(self, stride, padding=1):
        super(ExpandMask, self).__init__()
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        assert x.shape[1] == 1

        if self.stride > 1:
            self.pad_kernel = torch.zeros((1, 1, self.stride, self.stride), device=x.device)
            self.pad_kernel[0, 0, 0, 0] = 1
        self.dilate_kernel = torch.ones((1, 1, 1 + 2 * self.padding, 1 + 2 * self.padding), device=x.device)

        x = x.float()
        if self.stride > 1:
            x = F.conv_transpose2d(x, self.pad_kernel, stride=self.stride, groups=x.size(1))
        x = F.conv2d(x, self.dilate_kernel, padding=self.padding, stride=1)
        return x > 0.5
