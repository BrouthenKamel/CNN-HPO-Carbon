import torch
import torch.nn as nn
from torchvision import models

from src.loading.models.mobilenet.utils import make_divisible
from src.loading.models.mobilenet.config import InvertedResidualConfig, MobileNetConfig, original_config

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, squeeze_channels, activation_layer):
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels, squeeze_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_channels, in_channels, 1)
        self.activation = activation_layer()

    def forward(self, x):
        scale = self.fc1(x.mean((2, 3), keepdim=True))
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = self.activation(scale)
        return x * scale

class ConvBNActivation(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, norm_layer, activation_layer, stride=1, groups=1):
        
        padding = (kernel_size - 1) // 2
        
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, bias=False), norm_layer(out_channels), activation_layer(inplace=True)
        )

class InvertedResidual(nn.Module):
    def __init__(self, config: InvertedResidualConfig):
        
        super().__init__()
        
        self.use_res_connect = config.conv_bn_activation_config.stride == 1 and config.conv_bn_activation_config.in_channels == config.conv_bn_activation_config.out_channels

        layers = []

        if config.expanded_channels != config.conv_bn_activation_config.in_channels:
            layers.append(
                ConvBNActivation(config.conv_bn_activation_config.in_channels, config.expanded_channels, kernel_size=1, norm_layer=config.conv_bn_activation_config.norm_layer, activation_layer=config.conv_bn_activation_config.activation_layer)
            )

        layers.append(
            ConvBNActivation(config.expanded_channels, config.expanded_channels, kernel_size=config.conv_bn_activation_config.kernel_size, stride=config.conv_bn_activation_config.stride, groups=config.expanded_channels, norm_layer=config.conv_bn_activation_config.norm_layer, activation_layer=config.conv_bn_activation_config.activation_layer)
        )

        if config.use_se:
            squeeze_channels = make_divisible(config.expanded_channels // config.se_config.squeeze_factor, 8)
            layers.append(SqueezeExcitation(config.expanded_channels, squeeze_channels, config.se_config.activation_layer))

        layers.append(
            ConvBNActivation(config.expanded_channels, config.conv_bn_activation_config.out_channels, kernel_size=1, norm_layer=config.conv_bn_activation_config.norm_layer, activation_layer=config.conv_bn_activation_config.activation_layer)
        )

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        result = self.block(x)
        if self.use_res_connect:
            result = result + x
        return result

class MobileNetV3Small(nn.Module):
    
    def __init__(self, config: MobileNetConfig = original_config, num_classes=1000, weights=None):
        super().__init__()

        layers = []

        layers.append(
            ConvBNActivation(config.initial_conv_config.in_channels, config.initial_conv_config.out_channels, kernel_size=config.initial_conv_config.kernel_size, stride=config.initial_conv_config.stride, norm_layer=config.initial_conv_config.norm_layer, activation_layer=config.initial_conv_config.activation_layer)
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
