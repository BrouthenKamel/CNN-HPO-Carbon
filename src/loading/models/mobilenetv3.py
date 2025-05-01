import torch
import torch.nn as nn
from functools import partial

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class SqueezeExcitation(nn.Module):
    def __init__(self, input_channels, squeeze_channels):
        super().__init__()
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)
        self.hsigmoid = nn.Hardsigmoid(inplace=True)

    def forward(self, x):
        scale = self.fc1(x.mean((2, 3), keepdim=True))
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = self.hsigmoid(scale)
        return x * scale

class ConvBNActivation(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None, activation_layer=None, dilation=1):
        padding = (kernel_size - 1) // 2 * dilation
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU

        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=False),
            norm_layer(out_planes),
            activation_layer(inplace=True)
        )

class InvertedResidual(nn.Module):
    def __init__(self, config, norm_layer):
        super().__init__()
        self.use_res_connect = config.stride == 1 and config.input_channels == config.out_channels

        layers = []
        activation_layer = nn.Hardswish if config.use_hs else nn.ReLU

        if config.expanded_channels != config.input_channels:
            layers.append(
                ConvBNActivation(config.input_channels, config.expanded_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=activation_layer)
            )

        layers.append(
            ConvBNActivation(config.expanded_channels, config.expanded_channels, kernel_size=config.kernel, stride=config.stride,
                             groups=config.expanded_channels, norm_layer=norm_layer, activation_layer=activation_layer, dilation=config.dilation)
        )

        if config.use_se:
            squeeze_channels = _make_divisible(config.expanded_channels // 4, 8)
            layers.append(SqueezeExcitation(config.expanded_channels, squeeze_channels))

        layers.append(
            ConvBNActivation(config.expanded_channels, config.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None)
        )

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        result = self.block(x)
        if self.use_res_connect:
            result = result + x
        return result

class InvertedResidualConfig:
    def __init__(self, input_c, kernel, expand_c, out_c, use_se, act, stride, dilation, width_mult):
        self.input_channels = _make_divisible(input_c * width_mult, 8)
        self.kernel = kernel
        self.expanded_channels = _make_divisible(expand_c * width_mult, 8)
        self.out_channels = _make_divisible(out_c * width_mult, 8)
        self.use_se = use_se
        self.use_hs = act == "HS"
        self.stride = stride
        self.dilation = dilation

class MobileNetV3Small(nn.Module):
    def __init__(self, num_classes=10, width_mult=1.0, reduced_tail=False, dilated=False, weights=None):
        super().__init__()
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
        reduce_divider = 2 if reduced_tail else 1
        dilation = 2 if dilated else 1

        bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
        adjust_channels = partial(_make_divisible, divisor=8)

        # Configuration copied from torchvision
        layers_config = [
            bneck_conf(16, 3, 16, 16, True, "RE", 2, 1),
            bneck_conf(16, 3, 72, 24, False, "RE", 2, 1),
            bneck_conf(24, 3, 88, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 96, 40, True, "HS", 2, 1),
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 120, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 144, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2, dilation),
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
        ]

        layers = []
        firstconv_out = layers_config[0].input_channels
        layers.append(
            ConvBNActivation(3, firstconv_out, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=nn.Hardswish)
        )

        for config in layers_config:
            layers.append(InvertedResidual(config, norm_layer))

        lastconv_input = layers_config[-1].out_channels
        lastconv_output = 6 * lastconv_input
        layers.append(
            ConvBNActivation(lastconv_input, lastconv_output, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.Hardswish)
        )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        last_channel = adjust_channels(1024 // reduce_divider)
        self.classifier = nn.Sequential(
            nn.Linear(lastconv_output, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(last_channel, num_classes)
        )

        if weights is None:
            # self._initialize_weights()
            pass
        else:
            
            self.load_state_dict(weights.get_state_dict())

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
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

if __name__ == "__main__":
    
    from torchvision import models
    from torchvision.models import mobilenet_v3_small
    
    # Get pretrained weights from torchvision
    tv_model = mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    custom_model = MobileNetV3Small()
    
    custom_model_w = MobileNetV3Small(weights=tv_model.state_dict())

    # Transfer weights
    custom_model.load_state_dict(tv_model.state_dict())
    print("Loaded weights successfully!")
