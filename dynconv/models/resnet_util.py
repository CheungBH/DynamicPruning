import torch
import torch.nn as nn
from .pretrain_mask import *
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
import dynconv
from dynconv.maskunit import StatMaskUnit
import models.resnet_util
from models.channel_saliency import conv_forward, bn_relu_foward, channel_process, ChannelVectorUnit, GumbelChannelUnit


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    """Standard residual block """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, sparse=False, mask_type="conv"):
        super(BasicBlock, self).__init__()
        assert groups == 1
        assert dilation == 1
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.sparse = sparse

        if sparse:
            if mask_type == "conv":
            # in the resnet basic block, the first convolution is already strided, so mask_stride = 1
                self.masker = dynconv.MaskUnit(channels=inplanes, stride=stride, dilate_stride=1)
            elif mask_type == "stat":
                self.masker = StatMaskUnit(stride=stride, dilate_stride=1)
            else:
                raise NotImplementedError

        self.fast = False

    def forward(self, input):
        x, meta = input
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        if not self.sparse:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            out += identity
        else:
            assert meta is not None
            m = self.masker(x, meta)
            mask_dilate, mask = m['dilate'], m['std']

            x = dynconv.conv3x3(self.conv1, x, None, mask_dilate)
            x = dynconv.bn_relu(self.bn1, self.relu, x, mask_dilate)
            x = dynconv.conv3x3(self.conv2, x, mask_dilate, mask)
            x = dynconv.bn_relu(self.bn2, None, x, mask)
            out = identity + dynconv.apply_mask(x, mask)

        out = self.relu(out)
        return out, meta


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1,
                 norm_layer=None, sparse=False, resolution_mask=False, mask_block=False, mask_type="conv",
                 save_feat=False, input_resolution=False, conv1_act="relu", channel_budget=-1, channel_unit_type="fc",
                 group_size=1, dropout_ratio=0, dropout_stages=[-1], budget=-1, before_residual=False, **kwargs):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        print(f'Bottleneck - sparse: {sparse}: inp {inplanes}, hidden_dim {width}, ' + 
              f'oup {planes * self.expansion}, stride {stride}')

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        if conv1_act == "relu":
            self.conv1_act = nn.ReLU(inplace=True)
        elif conv1_act == "leaky_relu":
            self.conv1_act = nn.LeakyReLU(inplace=True)
        elif conv1_act == "none":
            self.conv1_act = None
        else:
            raise NotImplementedError
        self.downsample = downsample
        self.stride = stride
        self.sparse = sparse
        self.resolution_mask = resolution_mask
        self.mask_block = mask_block
        self.save_feat = save_feat
        self.mask_type = mask_type
        self.input_resolution = input_resolution
        self.mask_sampler = nn.MaxPool2d(kernel_size=2)
        self.channel_budget = channel_budget
        self.spatial_budget = budget
        self.dropout_ratio = dropout_ratio
        self.dropout_stages = dropout_stages
        self.before_residual = before_residual

        if sparse:
            if channel_budget >= 0:
                if channel_unit_type == "fc":
                    self.saliency = ChannelVectorUnit(in_channels=inplanes, out_channels=planes,
                                                      group_size=group_size, channel_budget=channel_budget, **kwargs)
                elif channel_unit_type == "fc_gumbel":
                    self.saliency = GumbelChannelUnit(inplanes=inplanes, outplanes=planes, group_size=group_size,
                                                      budget=channel_budget, **kwargs)
                else:
                    raise NotImplementedError

            if resolution_mask and not self.mask_block:
                pass
            else:
                if mask_type == "conv":
                    # in the resnet basic block, the first convolution is already strided, so mask_stride = 1
                    if not input_resolution:
                        self.masker = dynconv.MaskUnit(channels=inplanes, stride=stride, dilate_stride=1,
                                                       budget=self.spatial_budget, **kwargs)
                    else:
                        self.masker = dynconv.MaskUnit(channels=inplanes, stride=1, dilate_stride=1,
                                                       budget=self.spatial_budget, input_resolution=True, **kwargs)
                elif mask_type == "stat":
                    raise NotImplementedError
                    self.masker = dynconv.StatMaskUnit(stride=stride, dilate_stride=1)
                elif mask_type == "stat_mom":
                    self.masker = dynconv.StatMaskUnitMom(stride=stride, dilate_stride=1, budget=self.spatial_budget,
                                                          **kwargs)
                else:
                    raise NotImplementedError
        else:
            if mask_type == "none":
                self.mask = NoneMask()
            elif mask_type == "conv" and budget == -1:
                self.mask = NoneMask()
            elif mask_type == "zero_ratio":
                self.mask = ZeroRatioMask(**kwargs)
            elif mask_type == "sum":
                self.mask = SumNormalizeMask(**kwargs)
            elif mask_type == "zero_top":
                self.mask = ZeroRatioTopMask(**kwargs)
            elif mask_type == "random":
                self.mask = RandomMask(**kwargs)
            elif mask_type == "abs_sum":
                self.mask = AbsSumNormalizeMask(**kwargs)
            else:
                raise NotImplementedError("Unregistered mask type!")

    def forward_conv(self, x, conv1_mask=None):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if conv1_mask is not None:
            out = out * conv1_mask.unsqueeze(dim=1)
            if self.conv2.stride[0] == 2:
                conv1_mask = self.mask_sampler(conv1_mask)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        return out, conv1_mask

    def obtain_mask(self, x, meta):
        if self.resolution_mask:
            if self.mask_block:
                m = self.masker(x, meta)
            else:
                if self.input_resolution:
                    m = {"dilate": meta["masks"][-1]["std"], "std": meta["masks"][-1]["std"]}
                else:
                    m = meta["masks"][-1]
        else:
            m = self.masker(x, meta)
        return m

    def add_dropout(self, x, meta):
        if meta["stage_id"] in self.dropout_stages and 0 < self.dropout_ratio < 1:
            # test = (torch.rand_like(x) > self.dropout_ratio).int()
            return (torch.rand_like(x) > self.dropout_ratio).int() * x
        else:
            return x

    def forward(self, input):
        x, meta = input
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        if not self.sparse:
            if not isinstance(self.mask, NoneMask):
                if self.input_resolution:
                    mask = self.mask.process(x, 1,  meta["stage_id"])
                    out, mask = self.forward_conv(x, mask)
                else:
                    mask = self.mask.process(x, self.conv2.stride[0] != 1,  meta["stage_id"])
                    out, _ = self.forward_conv(x, mask)
                out = out * mask.unsqueeze(dim=1)
            else:
                out, _ = self.forward_conv(x)

            if self.save_feat:
                meta["feat_before"].append(out)
            out += identity
            if self.save_feat:
                meta["feat_after"].append(out)
        else:
            assert meta is not None
            meta["stride"] = self.stride
            if self.channel_budget > 0:
                vector, meta = self.saliency(x, meta)
                conv_forward(self.conv1, None, None, vector, forward=False)
                conv_forward(self.conv2, None, vector, vector, forward=False)
                conv_forward(self.conv3, None, vector, None, forward=False)

            m = self.obtain_mask(self.add_dropout(x.clone(), meta), meta)
            mask_dilate, mask = m['dilate'], m['std']

            x = dynconv.conv1x1(self.conv1, x, mask_dilate)
            x = dynconv.bn_relu(self.bn1, self.conv1_act, x, mask_dilate)
            x = dynconv.apply_mask(x, mask_dilate) if self.input_resolution else x
            if self.channel_budget > 0:
                x = channel_process(x, vector)
            x = dynconv.conv3x3(self.conv2, x, mask_dilate, mask)
            x = dynconv.bn_relu(self.bn2, self.relu, x, mask)
            if self.channel_budget > 0:
                x = channel_process(x, vector)

            x = dynconv.conv1x1(self.conv3, x, mask)
            x = dynconv.bn_relu(self.bn3, None, x, mask)
            # meta["saliency_mask"] = self.get_saliency_mask(x, mask.hard)
            out = identity + dynconv.apply_mask(x, mask)
            meta["saliency_mask"] = x if self.before_residual else out
        meta["block_id"] += 1
        out = self.relu(out)
        return out, meta

    def get_saliency_mask(self, x, mask=None):
        if mask is None:
            return torch.ones(x.shape[0], 1, x.shape[-2], x.shape[-1]).cuda()
        else:
            return mask

