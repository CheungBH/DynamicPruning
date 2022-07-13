import dynconv
import torch
from models.channel_saliency import conv_forward, bn_relu_foward, channel_process, ChannelVectorUnit, GumbelChannelUnit
from torch import nn


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None, sparse=False):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None, sparse=False, resolution_mask=False,
                 mask_block=False, mask_type="conv", final_activation="linear", downsample=False,
                 input_resolution=False, dropout_ratio=0, dropout_stages=[-1], channel_budget=-1,
                 channel_unit_type="fc", budget=-1, group_size=1, **kwargs):
        super(InvertedResidualBlock, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.sparse = sparse
        self.channel_budget = channel_budget
        self.spatial_budget = budget
        self.use_res_connect = (self.stride == 1 and inp == oup) if downsample is None else True
        self.downsample = downsample

        self.resolution_mask = resolution_mask
        self.mask_block = mask_block
        self.input_resolution = input_resolution
        hidden_dim = int(round(inp * expand_ratio))

        if sparse:
            if channel_budget >= 0:
                if channel_unit_type == "fc":
                    self.saliency = ChannelVectorUnit(in_channels=inp, out_channels=hidden_dim,
                                                      group_size=group_size, channel_budget=channel_budget, **kwargs)
                elif channel_unit_type == "fc_gumbel":
                    self.saliency = GumbelChannelUnit(inplanes=inp, outplanes=hidden_dim, group_size=group_size,
                                                      budget=channel_budget, **kwargs)

                else:
                    raise NotImplementedError

        if self.sparse and self.use_res_connect:
            if self.resolution_mask:
                if self.mask_block:
                    self.masker = dynconv.MaskUnit(channels=inp, stride=stride, budget=budget, dilate_stride=1)
            else:
                if mask_type == "conv":
                    if not input_resolution:
                        self.masker = dynconv.MaskUnit(channels=inp, stride=stride, budget=budget, dilate_stride=1,
                                                       **kwargs)
                    else:
                        self.masker = dynconv.MaskUnit(channels=inp, stride=1, budget=budget, dilate_stride=1,
                                                       input_resolution=True, **kwargs)
                elif mask_type == "stat":
                    raise NotImplementedError
                    self.masker = dynconv.StatMaskUnit(stride=stride, dilate_stride=1)
                elif mask_type == "stat_mom":
                    self.masker = dynconv.StatMaskUnitMom(stride=stride, budget=budget, dilate_stride=1, **kwargs)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.activation = nn.ReLU(inplace=True)

        self.conv_pw_1 = nn.Conv2d(inp, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(hidden_dim)

        self.conv3x3_dw = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim,
                                    bias=False)
        self.bn_dw = norm_layer(hidden_dim)

        self.conv_pw_2 = nn.Conv2d(hidden_dim, oup, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = norm_layer(oup)
        self.final_activation = final_activation

        self.dropout_ratio = dropout_ratio
        self.dropout_stages = dropout_stages

    def forward_basic(self, x):
        x = self.activation(self.bn1(self.conv_pw_1(x)))
        x = self.activation(self.bn_dw(self.conv3x3_dw(x)))
        x = self.bn2(self.conv_pw_2(x))
        return x

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

    def forward_channel_pruning(self, x, meta):

        vector, meta = self.saliency(x, meta)

        conv_forward(self.conv_pw_1, None, None, vector, forward=False)
        conv_forward(self.conv3x3_dw, None, None, vector, forward=False)
        conv_forward(self.conv_pw_2, None, vector, None, forward=False)

        x = self.activation(self.bn1(self.conv_pw_1(x)))
        x = channel_process(x, vector)
        x = self.activation(self.bn_dw(self.conv3x3_dw(x)))
        x = channel_process(x, vector)
        x = self.bn2(self.conv_pw_2(x))
        meta["saliency_mask"] = None

        return x

    def forward_block(self, inp):
        x, meta = inp
        if (not self.sparse) or (not self.use_res_connect):
            if self.channel_budget == -1:
                x = self.forward_basic(x)
            else:
                x = self.forward_channel_pruning(x, meta)
        else:
            meta["stride"] = self.stride
            m = self.obtain_mask(x, meta)
            mask_dilate, mask = m['dilate'], m['std']
            if self.channel_budget > 0:
                vector, meta = self.saliency(x, meta)
                conv_forward(self.conv_pw_1, None, None, vector, forward=False)
                conv_forward(self.conv3x3_dw, None, None, vector, forward=False)
                conv_forward(self.conv_pw_2, None, vector, None, forward=False)

            x = dynconv.conv1x1(self.conv_pw_1, x, mask, mask_dilate)
            x = dynconv.bn_relu(self.bn1, self.activation, x, mask_dilate)
            x = dynconv.apply_mask(x, mask_dilate) if self.input_resolution else x
            if self.channel_budget > 0:
                x = channel_process(x, vector)
            x = dynconv.conv3x3_dw(self.conv3x3_dw, x, mask_dilate, mask)
            x = dynconv.bn_relu(self.bn_dw, self.activation, x, mask)
            if self.channel_budget > 0:
                x = channel_process(x, vector)
            x = dynconv.conv1x1(self.conv_pw_2, x, mask_dilate, mask)
            x = dynconv.bn_relu(self.bn2, None, x, mask)
            meta["saliency_mask"] = self.get_masked_feature(x, mask.hard)
            x = dynconv.apply_mask(x, mask)
        return (x, meta)

    def forward(self, inp):
        x, meta = inp
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.forward_block(inp)
        meta["stage_id"], meta["block_id"] = meta["stage_id"] + 1, meta["block_id"] + 1
        if self.final_activation == "linear":
            return (identity + out[0], out[1]) if self.use_res_connect else out
        elif self.final_activation == "relu":
            return (self.activation(identity + out[0]), out[1]) if self.use_res_connect \
                else (self.activation(out[0]), out[1])
        else:
            raise NotImplementedError

    def get_masked_feature(self, x, mask=None):
        if mask is None:
            return x
        else:
            return mask.float().expand_as(x) * x
