import torch
import torch.nn as nn
from torchvision import models
from functools import partial
from typing import List

def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class SqueezeExcitation(nn.Module): # squeeze channels, activation_layer
    def __init__(self, in_channels, squeeze_channels, activation_layer): # Hardsigmoid, Sigmoid
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels, squeeze_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_channels, in_channels, 1)
        self.activation = activation_layer(inplace=True)

    def forward(self, x):
        scale = self.fc1(x.mean((2, 3), keepdim=True))
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = self.activation(scale)
        return x * scale

class ConvBNActivation(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, norm_layer, activation_layer, stride=1, padding=0, groups=1):
        
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, bias=False), norm_layer(out_channels), activation_layer(inplace=True)
        )
        
class SqueezeExcitationConfig:
    def __init__(self, squeeze_factor: int = 4, activation_layer: str = "Hardsigmoid"):
        self.squeeze_factor = squeeze_factor
        self.activation_layer = self.get_activation_layer(activation_layer)
        
    def get_activation_layer(self, activation_layer):
        if activation_layer == "Hardsigmoid":
            return nn.Hardsigmoid
        elif activation_layer == "Sigmoid":
            return nn.Sigmoid
        else:
            raise ValueError(f"Unsupported activation layer: {self.activation_layer}")
        
class ConvBNActivationConfig:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int,
                 norm_layer: str = "BatchNorm2d", activation_layer: str = "ReLU", eps: float = None, momentum: float = None, ignore_in_channels: bool = False):
        if ignore_in_channels:
            self.in_channels = in_channels
        else:
            self.in_channels = _make_divisible(in_channels, 8)
        self.out_channels = _make_divisible(out_channels, 8)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.norm_layer = self.get_norm_layer(norm_layer, eps, momentum)
        
        self.activation_layer = self.get_activation_layer(activation_layer)
        
    def get_norm_layer(self, norm_layer, eps, momentum):
        if norm_layer == "BatchNorm2d":
            kwargs = {}
            if eps != None:
                kwargs['eps'] = eps
            if momentum != None:
                kwargs['momentum'] = momentum
            return partial(nn.BatchNorm2d, **kwargs)
        else:
            raise ValueError(f"Unsupported normalization layer: {self.norm_layer}")
        
    def get_activation_layer(self, activation_layer):
        if activation_layer == "ReLU":
            return nn.ReLU
        elif activation_layer == "Hardswish":
            return nn.Hardswish
        else:
            raise ValueError(f"Unsupported activation layer: {self.activation_layer}")

class InvertedResidualConfig:
    def __init__(self, expand_channels: int, use_se: bool, se_config: SqueezeExcitationConfig, conv_bn_activation_config: ConvBNActivationConfig):
        self.use_se = use_se
        self.se_config = se_config
        self.conv_bn_activation_config = conv_bn_activation_config
        self.expanded_channels = _make_divisible(expand_channels, 8)

class InvertedResidual(nn.Module):
    def __init__(self, config: InvertedResidualConfig):
        
        super().__init__()
        
        self.use_res_connect = config.conv_bn_activation_config.stride == 1 and config.conv_bn_activation_config.in_channels == config.conv_bn_activation_config.out_channels

        layers = []

        if config.expanded_channels != config.conv_bn_activation_config.in_channels:
            layers.append(
                ConvBNActivation(config.conv_bn_activation_config.in_channels, config.expanded_channels, kernel_size=1, norm_layer=config.conv_bn_activation_config.norm_layer, activation_layer=config.conv_bn_activation_config.activation_layer, stride=1, padding=0)
            )

        layers.append(
            ConvBNActivation(config.expanded_channels, config.expanded_channels, kernel_size=config.conv_bn_activation_config.kernel_size, stride=config.conv_bn_activation_config.stride, padding=config.conv_bn_activation_config.padding, groups=config.expanded_channels, norm_layer=config.conv_bn_activation_config.norm_layer, activation_layer=config.conv_bn_activation_config.activation_layer)
        )

        if config.use_se:
            squeeze_channels = _make_divisible(config.expanded_channels // config.se_config.squeeze_factor, 8)
            layers.append(SqueezeExcitation(config.expanded_channels, squeeze_channels, config.se_config.activation_layer))

        layers.append(
            ConvBNActivation(config.expanded_channels, config.conv_bn_activation_config.out_channels, kernel_size=1, norm_layer=config.conv_bn_activation_config.norm_layer, activation_layer=config.conv_bn_activation_config.activation_layer, stride=1, padding=0)
        )

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        result = self.block(x)
        if self.use_res_connect:
            result = result + x
        return result

class ClassifierConfig:
    def __init__(self, neurons: int, activation_layer: str, dropout_rate: float):
        
        self.neurons = neurons
        self.activation_layer = self.get_activation_layer(activation_layer)
        self.dropout_rate = dropout_rate
    
    def get_activation_layer(self, activation_layer):
        if activation_layer == "ReLU":
            return nn.ReLU
        elif activation_layer == "Hardswish":
            return nn.Hardswish
        else:
            raise ValueError(f"Unsupported activation layer: {self.activation_layer}")

class MobileNetConfig:
    def __init__(self, initial_conv_config: ConvBNActivationConfig, last_conv_upsample: int, last_conv_config: ConvBNActivationConfig, inverted_residual_configs: list[InvertedResidualConfig], classifier_config: ClassifierConfig):
        
        self.initial_conv_config = initial_conv_config
        self.inverted_residual_configs = inverted_residual_configs
        self.last_conv_upsample = last_conv_upsample
        self.last_conv_config = last_conv_config
        self.classifier_config = classifier_config

class MobileNetV3Small(nn.Module):
    
    def __init__(self, num_classes=1000, weights=None):
        super().__init__()
        
        inverted_residual_configs = [
                        
            InvertedResidualConfig(
                use_se = True,
                se_config = SqueezeExcitationConfig(squeeze_factor=4, activation_layer="Hardsigmoid"),
                conv_bn_activation_config = ConvBNActivationConfig(
                    in_channels=16,
                    out_channels=16,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    norm_layer="BatchNorm2d",
                    activation_layer="ReLU"
                ),
                expand_channels=16,
            ),
                        
            InvertedResidualConfig(
                use_se = False,
                se_config = SqueezeExcitationConfig(squeeze_factor=4, activation_layer="Hardsigmoid"),
                conv_bn_activation_config = ConvBNActivationConfig(
                    in_channels=16,
                    out_channels=24,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    norm_layer="BatchNorm2d",
                    activation_layer="ReLU"
                ),
                expand_channels=72,
            ),
                        
            InvertedResidualConfig(
                use_se = False,
                se_config = SqueezeExcitationConfig(squeeze_factor=4, activation_layer="Hardsigmoid"),
                conv_bn_activation_config = ConvBNActivationConfig(
                    in_channels=24,
                    out_channels=24,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_layer="BatchNorm2d",
                    activation_layer="Hardswish"
                ),
                expand_channels=88,
            ),
                        
            InvertedResidualConfig(
                use_se = True,
                se_config = SqueezeExcitationConfig(squeeze_factor=4, activation_layer="Hardsigmoid"),
                conv_bn_activation_config = ConvBNActivationConfig(
                    in_channels=24,
                    out_channels=40,
                    kernel_size=5,
                    stride=2,
                    padding=2,
                    norm_layer="BatchNorm2d",
                    activation_layer="Hardswish"
                ),
                expand_channels=96,
            ),
                        
            InvertedResidualConfig(
                use_se = True,
                se_config = SqueezeExcitationConfig(squeeze_factor=4, activation_layer="Hardsigmoid"),
                conv_bn_activation_config = ConvBNActivationConfig(
                    in_channels=40,
                    out_channels=40,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    norm_layer="BatchNorm2d",
                    activation_layer="Hardswish"
                ),
                expand_channels=240,
            ),
            
            InvertedResidualConfig(
                use_se = True,
                se_config = SqueezeExcitationConfig(squeeze_factor=4, activation_layer="Hardsigmoid"),
                conv_bn_activation_config = ConvBNActivationConfig(
                    in_channels=40,
                    out_channels=40,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    norm_layer="BatchNorm2d",
                    activation_layer="Hardswish"
                ),
                expand_channels=240,
            ),
            
            InvertedResidualConfig(
                use_se = True,
                se_config = SqueezeExcitationConfig(squeeze_factor=4, activation_layer="Hardsigmoid"),
                conv_bn_activation_config = ConvBNActivationConfig(
                    in_channels=40,
                    out_channels=48,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    norm_layer="BatchNorm2d",
                    activation_layer="Hardswish"
                ),
                expand_channels=120,
            ),
            
            InvertedResidualConfig(
                use_se = True,
                se_config = SqueezeExcitationConfig(squeeze_factor=4, activation_layer="Hardsigmoid"),
                conv_bn_activation_config = ConvBNActivationConfig(
                    in_channels=48,
                    out_channels=48,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    norm_layer="BatchNorm2d",
                    activation_layer="Hardswish"
                ),
                expand_channels=144,
            ),
            
            InvertedResidualConfig(
                use_se = True,
                se_config = SqueezeExcitationConfig(squeeze_factor=4, activation_layer="Hardsigmoid"),
                conv_bn_activation_config = ConvBNActivationConfig(
                    in_channels=48,
                    out_channels=96,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    norm_layer="BatchNorm2d",
                    activation_layer="Hardswish"
                ),
                expand_channels=288,
            ),
            
            InvertedResidualConfig(
                use_se = True,
                se_config = SqueezeExcitationConfig(squeeze_factor=4, activation_layer="Hardsigmoid"),
                conv_bn_activation_config = ConvBNActivationConfig(
                    in_channels=96,
                    out_channels=96,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    norm_layer="BatchNorm2d",
                    activation_layer="Hardswish"
                ),
                expand_channels=576,
            ),
                        
            InvertedResidualConfig(
                use_se = True,
                se_config = SqueezeExcitationConfig(squeeze_factor=4, activation_layer="Hardsigmoid"),
                conv_bn_activation_config = ConvBNActivationConfig(
                    in_channels=96,
                    out_channels=96,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    norm_layer="BatchNorm2d",
                    activation_layer="Hardswish"
                ),
                expand_channels=576,
            ),
        ]
        
        last_conv_upsample = 6
        
        config = MobileNetConfig(
            initial_conv_config=ConvBNActivationConfig(
                in_channels=3,
                out_channels=inverted_residual_configs[0].conv_bn_activation_config.in_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                activation_layer='Hardswish',
                norm_layer='BatchNorm2d',
                eps=1e-3,
                momentum=1e-2,
                ignore_in_channels=True
            ),
            inverted_residual_configs=inverted_residual_configs,
            last_conv_upsample=last_conv_upsample,
            last_conv_config=ConvBNActivationConfig(
                in_channels=inverted_residual_configs[-1].conv_bn_activation_config.out_channels,
                out_channels=inverted_residual_configs[-1].conv_bn_activation_config.out_channels * last_conv_upsample,
                kernel_size=1,
                stride=1,
                padding=0,
                activation_layer='Hardswish',
                norm_layer='BatchNorm2d',
                eps=1e-3,
                momentum=1e-2
            ),
            classifier_config=ClassifierConfig(
                neurons=1024,
                activation_layer='Hardswish',
                dropout_rate=0.2
            )
        )

        layers = []

        layers.append(
            ConvBNActivation(config.initial_conv_config.in_channels, config.initial_conv_config.out_channels, kernel_size=config.initial_conv_config.kernel_size, stride=config.initial_conv_config.stride, padding=config.initial_conv_config.padding, norm_layer=config.initial_conv_config.norm_layer, activation_layer=config.initial_conv_config.activation_layer)
        )

        for inverted_residual_config in config.inverted_residual_configs:
            layers.append(InvertedResidual(inverted_residual_config))
        
        layers.append(
            ConvBNActivation(config.last_conv_config.in_channels, config.last_conv_config.out_channels, kernel_size=config.last_conv_config.kernel_size, norm_layer=config.last_conv_config.norm_layer, activation_layer=config.last_conv_config.activation_layer)
        )

        self.features = nn.Sequential(*layers)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(config.last_conv_config.out_channels, config.classifier_config.neurons),
            config.classifier_config.activation_layer(inplace=True),
            nn.Dropout(p=config.classifier_config.dropout_rate, inplace=True),
            nn.Linear(config.classifier_config.neurons, num_classes)
        )

        if weights is not None:
            self.load_state_dict(weights.get_state_dict())

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    
    custom_model = MobileNetV3Small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    
    print("Loaded weights successfully!")
