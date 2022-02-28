from torch import nn
import dynconv


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
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None, sparse=False):
        super(InvertedResidualBlock, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.sparse = sparse
        if self.sparse:
            self.masker = dynconv.MaskUnit(channels=inp, stride=stride, dilate_stride=1)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        self.squeeze = False
        self.activation = nn.ReLU(inplace=True)

        if expand_ratio != 1:
            self.squeeze = True
            self.conv_pw_1 = nn.Conv2d(inp, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn1 = norm_layer(hidden_dim)

        self.conv3x3_dw = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False)
        self.bn_dw = norm_layer(hidden_dim)

        self.conv_pw_2 = nn.Conv2d(hidden_dim, oup, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = norm_layer(oup)

    def forward_basic(self, inp):
        x, meta = inp
        if not self.sparse:
            if self.squeeze:
                x = self.activation(self.bn1(self.conv_pw_1(x)))
            x = self.activation(self.bn_dw(self.conv3x3_dw(x)))
            x = self.bn2(self.conv_pw_2(x))
        else:
            m = self.masker(x, meta)
            mask_dilate, mask = m['dilate'], m['std']

            if self.squeeze:
                x = dynconv.conv1x1(self.conv_pw_1, x, mask, mask_dilate)
                x = dynconv.bn_relu(self.bn1, self.activation, x, mask_dilate)
            x = dynconv.conv3x3_dw(self.conv3x3_dw, x, mask_dilate, mask)
            x = dynconv.bn_relu(self.bn_dw, self.activation, x, mask)
            x = dynconv.conv1x1(self.conv_pw_2, x, mask_dilate, mask)
            x = dynconv.bn_relu(self.bn2, None, x, mask)
            x = dynconv.apply_mask(x, mask)
        return (x, meta)

    def forward(self, inp):
        x, meta = inp
        out = self.forward_basic(inp)
        if self.use_res_connect:
            return (x + out[0], out[1])
        else:
            return out


class MobileNetV2_32x32(nn.Module):
    def __init__(self,
                 num_classes=10,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 sparse=False,
                 round_nearest=8,
                 block=None,
                 norm_layer=None):
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
                                      sparse=sparse))
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
        x = self.first_conv(x)
        x, meta = self.features((x, meta))
        x = self.final_conv(x)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x, meta

