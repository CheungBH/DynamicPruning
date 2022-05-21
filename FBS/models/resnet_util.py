import torch
import torch.nn as nn
from .pretrain_mask import *
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
import models.resnet_util
from models.channel_saliency import *


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
                 base_width=64, dilation=1, norm_layer=None, sparse=False, channel_saliency="conv"):
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
            if channel_saliency == "fc":
                self.masker = ChannelVectorUnit(channels=inplanes, stride=stride, dilate_stride=1)
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

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            out += identity

        out = self.relu(out)
        return out, meta

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1,
                 norm_layer=None, sparse=False, resolution_mask=False, mask_block=False, mask_type="conv",
                 save_feat=False, input_resolution=False, group_size=1, budget=-1, **kwargs):
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
        self.downsample = downsample
        self.stride = stride
        self.sparse = sparse
        self.budget = budget
        self.resolution_mask = resolution_mask
        self.mask_block = mask_block
        self.save_feat = save_feat
        self.mask_type = mask_type
        self.input_resolution = input_resolution
        self.mask_sampler = nn.MaxPool2d(kernel_size=2)

        if sparse:
            if resolution_mask and not self.mask_block:
                return
            else:
                if mask_type == "fc":
                    if not input_resolution:
                        self.saliency = ChannelVectorUnit(in_channels=inplanes, out_channels=planes,
                                                          group_size=group_size, budget=self.budget, **kwargs)
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError
        else:
            if mask_type == "none":
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

    def forward_conv(self, x, conv1_mask):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if conv1_mask:
            out = out * conv1_mask.unsqueeze(dim=1)
            if self.conv2.stride[0] == 2:
                conv1_mask = self.mask_sampler(conv1_mask)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        return out, conv1_mask

    def obtain_vector(self, meta):
        if self.resolution_mask:
            raise NotImplementedError
            # if self.mask_block:
            #     vector = self.saliency(x, meta)
            # else:
            #     if self.input_resolution:
            #         m = {"dilate": meta["masks"][-1]["std"], "std": meta["masks"][-1]["std"]}
            #     else:
            #         m = meta["masks"][-1]
        else:
            vector = self.saliency(meta["masked_feat"], meta)
        return vector

    def forward(self, input):
        x, meta = input
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        if not self.sparse:
            if self.input_resolution:
                mask = self.mask.process(x, 1,  meta["stage_id"])
                out, mask = self.forward_conv(x, mask)
            else:
                mask = self.mask.process(x, self.conv2.stride[0] != 1,  meta["stage_id"])
                out, _ = self.forward_conv(x, mask)
            out = out * mask.unsqueeze(dim=1)
            meta["block_id"] += 1

            if self.save_feat:
                meta["feat_before"].append(out)
            out += identity
            if self.save_feat:
                meta["feat_after"].append(out)
        else:
            assert meta is not None
            # meta["stride"] = self.stride
            vector = self.obtain_vector(meta)
            # meta["lasso_sum"] += torch.mean(torch.sum(vector, dim=-1))
            # vector_mask = winner_take_all(vector.clone(), self.budget)
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.channel_process(out, vector)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)
            out = self.channel_process(out, vector)

            out = self.conv3(out)
            out = self.bn3(out)
            out += identity

        out = self.relu(out)
        meta["masked_feat"] = self.get_masked_feature(out)
        return out, meta

    def channel_process(self, x, vector):
        if len(vector.shape) != 2:
            return x * vector
        else:
            return x * vector.unsqueeze(-1).unsqueeze(-1).expand_as(x)

    def get_masked_feature(self, x, mask=None):
        if mask is None:
            return x
        else:
            raise NotImplementedError

    # def forward_stride_mask(self, x, mask):
    #     return dynconv.apply_mask(x, mask) if self.input_resolution else x
