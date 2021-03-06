from torch import nn
from models.channel_saliency import conv_forward, bn_relu_foward, channel_process, ChannelVectorUnit
import torch


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


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
                 input_resolution=False, dropout_ratio=0, dropout_stages=[-1], group_size=64, budget=-1, **kwargs):
        super(InvertedResidualBlock, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.sparse = sparse
        self.budget = budget
        self.use_res_connect = (self.stride == 1 and inp == oup) if downsample is None else True
        self.downsample = downsample
        hidden_dim = int(round(inp * expand_ratio))
        self.resolution_mask = resolution_mask
        self.mask_block = mask_block
        self.input_resolution = input_resolution
        if self.sparse:
            if mask_type == "fc":
                self.saliency = ChannelVectorUnit(in_channels=inp, out_channels=hidden_dim,
                                                  group_size=group_size, budget=self.budget, **kwargs)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.activation = nn.ReLU(inplace=True)

        self.conv_pw_1 = nn.Conv2d(inp, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(hidden_dim)

        self.conv3x3_dw = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False)
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

    # def add_dropout(self, x, meta):
    #     if meta["stage_id"] in self.dropout_stages and 0 < self.dropout_ratio < 1:
    #         return (torch.rand_like(x) > self.dropout_ratio).int() * x
    #     else:
    #         return x

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

    def forward_block(self, inp):
        x, meta = inp
        if (not self.sparse) or (not self.use_res_connect):
            x = self.forward_basic(x)
        else:
            vector = self.saliency(x, meta)

            x = conv_forward(self.conv_pw_1, x, out_vec=vector)
            x = bn_relu_foward(self.bn1, self.activation, x, vector)
            x = channel_process(x, vector)

            x = conv_forward(self.conv3x3_dw, x, inp_vec=vector, out_vec=vector)
            x = bn_relu_foward(self.bn_dw, self.activation, x, vector)
            x = channel_process(x, vector)

            x = conv_forward(self.conv_pw_2, x, inp_vec=vector)
            x = bn_relu_foward(self.bn2, None, x)

        return (x, meta)

    def forward(self, inp):
        x, meta = inp
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.forward_block(inp)
        meta["block_id"] += 1
        if self.final_activation == "linear":
            return (identity + out[0], out[1]) if self.use_res_connect else out
        elif self.final_activation == "relu":
            return (self.activation(identity + out[0]), out[1]) if self.use_res_connect \
                else (self.activation(out[0]), out[1])
        else:
            raise NotImplementedError


class MobileNetV2_32x32(nn.Module):
    def __init__(self,
                 num_classes=10,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 sparse=False,
                 round_nearest=8,
                 block=None,
                 norm_layer=None,
                 **kwargs):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use

        """
        super(MobileNetV2_32x32, self).__init__()

        if block is None:
            block = InvertedResidualBlock

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 480

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 1],
                [6, 32, 3, 2],
                [6, 48, 2, 2],
                [6, 64, 2, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        self.first_conv = nn.Sequential(*[ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)])

        features = []
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer,
                                      sparse=sparse, **kwargs))
                input_channel = output_channel
        # building last several layers
        # features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        self.final_conv = nn.Sequential(*[ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer)])
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x, meta):
        meta["block_id"] = 0
        x = self.first_conv(x)
        x, meta = self.features((x, meta))
        x = self.final_conv(x)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x, meta

