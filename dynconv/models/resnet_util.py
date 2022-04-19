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
                 save_feat=False, **kwargs):
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
        self.resolution_mask = resolution_mask
        self.mask_block = mask_block
        self.save_feat = save_feat

        if sparse:
            if resolution_mask and not self.mask_block:
                return
            else:
                if mask_type == "conv":
                    # in the resnet basic block, the first convolution is already strided, so mask_stride = 1
                    self.masker = dynconv.MaskUnit(channels=inplanes, stride=stride, dilate_stride=1, **kwargs)
                elif mask_type == "stat":
                    self.masker = dynconv.StatMaskUnit(stride=stride, dilate_stride=1)
                elif mask_type == "stat_mom":
                    self.masker = dynconv.StatMaskUnitMom(stride=stride, dilate_stride=1, **kwargs)
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
            else:
                raise NotImplementedError("Unregistered mask type!")

    def forward(self, input):
        x, meta = input
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        if not self.sparse:

            mask = self.mask.process(x, self.conv2.stride[0] != 1,  meta["stage_id"])
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)
            try:
                out = out * mask
            except:
                out * mask.unsqueeze(dim=1)
            meta["block_id"] += 1

            if self.save_feat:
                meta["feat_before"].append(out)
            out += identity
            if self.save_feat:
                meta["feat_after"].append(out)
        else:
            assert meta is not None
            if self.resolution_mask:
                if self.mask_block:
                    m = self.masker(x, meta)
                else:
                    m = meta["masks"][-1]
            else:
                m = self.masker(x, meta)
            mask_dilate, mask = m['dilate'], m['std']

            x = dynconv.conv1x1(self.conv1, x, mask_dilate)
            x = dynconv.bn_relu(self.bn1, self.relu, x, mask_dilate)
            x = dynconv.conv3x3(self.conv2, x, mask_dilate, mask)
            x = dynconv.bn_relu(self.bn2, self.relu, x, mask)
            x = dynconv.conv1x1(self.conv3, x, mask)
            x = dynconv.bn_relu(self.bn3, None, x, mask)
            out = identity + dynconv.apply_mask(x, mask)

        out = self.relu(out)
        return out, meta
