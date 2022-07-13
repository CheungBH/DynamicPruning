from torch import nn
from .mobilenet_utils import *

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


class MobileNetV2_32x32(nn.Module):
    def __init__(self,
                 num_classes=10,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 sparse=False,
                 round_nearest=8,
                 block=None,
                 norm_layer=None,
                 use_downsample=False,
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
                [6, 24, 2, 2],
                [6, 32, 3, 1],
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
        block_sum = sum([s[2] for s in inverted_residual_setting])
        if kwargs['channel_stage'][-1] == -1:
            if len(kwargs['channel_stage']) == 2:
                kwargs['channel_stage'] = [kwargs['channel_stage'][0], block_sum-1]
            else:
                raise ValueError("Not correct stage idx for -1")

        # building inverted residual blocks
        for stage_idx, (t, c, n, s) in enumerate(inverted_residual_setting):
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                mask_block = int((s == 2 and i == 1))
                if use_downsample and (s == 2 or input_channel != output_channel):
                    downsample = nn.Sequential(
                        nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=stride, bias=False),
                        norm_layer(output_channel),
                    )
                else:
                    downsample = None
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer,
                                      sparse=sparse, mask_block=mask_block, downsample=downsample, **kwargs))
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
        meta["block_id"], meta["stage_id"], meta["saliency_mask"] = 0, 0, x
        x, meta = self.features((x, meta))
        x = self.final_conv(x)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x, meta

