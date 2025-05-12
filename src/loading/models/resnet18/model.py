import torch
import torch.nn as nn
from typing import Type, List, Optional, Callable, Union

from src.loading.models.resnet18.config import (
    ConvLayerConfig,
    BasicBlockConfig,
    ResNetConfig,
    original_config 
)

class ConvLayer(nn.Sequential):
    """Creates a Conv2d -> BatchNorm2d -> Activation layer sequence."""
    def __init__(self, config: ConvLayerConfig):
        super().__init__(
            nn.Conv2d(
                config.in_channels,
                config.out_channels,
                kernel_size=config.kernel_size,
                stride=config.stride,
                padding=config.padding,
                bias=config.bias
            ),
            config.norm_layer(**config.norm_layer_args),
            config.activation_layer() 
        )

class BasicBlock(nn.Module):
    """ResNet Basic Block implementation."""
    expansion: int = 1

    def __init__(self, config: BasicBlockConfig):
        super().__init__()
        self.conv1 = ConvLayer(config.conv1_config)
        self.conv2 = nn.Conv2d(
            config.conv2_config.in_channels,
            config.conv2_config.out_channels,
            kernel_size=config.conv2_config.kernel_size,
            stride=config.conv2_config.stride,
            padding=config.conv2_config.padding,
            bias=config.conv2_config.bias
        )
        self.bn2 = config.conv2_config.norm_layer(**config.conv2_config.norm_layer_args)
        self.activation = config.activation_layer()

        self.downsample: Optional[nn.Module] = None
        if config.downsample_config:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    config.downsample_config.in_channels,
                    config.downsample_config.out_channels,
                    kernel_size=config.downsample_config.kernel_size,
                    stride=config.downsample_config.stride,
                    padding=config.downsample_config.padding,
                    bias=config.downsample_config.bias
                ),
                config.downsample_config.norm_layer(**config.downsample_config.norm_layer_args)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity 
        out = self.activation(out) 
        return out

class ResNet(nn.Module):
    """Generic ResNet model builder."""
    def __init__(self, config: ResNetConfig = original_config, num_classes: int = 1000):
        super().__init__()
        self._config = config 

        self.conv1 = ConvLayer(config.initial_conv_config)
        # in standard ResNet the first activation and BN are separate
        # self.bn1 = config.initial_batch_norm_layer(**config.initial_batch_norm_args)
        # self.activation = config.initial_activation_layer()
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if config.initial_maxpool else nn.Identity()
        # the ConvLayer includes BN and Activation we just need the optional MaxPool.
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if config.initial_maxpool else nn.Identity()
        current_channels = config.initial_conv_config.out_channels 

        self.stages = nn.ModuleList()
        for stage_cfg_list in config.stage_configs:
            stage_layers = []
            for block_cfg in stage_cfg_list:
                stage_layers.append(BasicBlock(block_cfg))
                current_channels = block_cfg.conv2_config.out_channels * BasicBlock.expansion
            self.stages.append(nn.Sequential(*stage_layers))

        self.avgpool = nn.AdaptiveAvgPool2d(config.adaptive_pool_output_size)

        classifier_layers = []
        in_features = config.classifier_config.in_features * BasicBlock.expansion 
        
        for hidden_neurons in config.classifier_config.neurons:
            classifier_layers.append(nn.Linear(in_features, hidden_neurons))
            classifier_layers.append(config.classifier_config.activation_layer())
            if config.classifier_config.dropout_rate > 0:
                classifier_layers.append(nn.Dropout(p=config.classifier_config.dropout_rate))
            in_features = hidden_neurons 

        classifier_layers.append(nn.Linear(in_features, num_classes))
        
        self.fc = nn.Sequential(*classifier_layers)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        # x = self.bn1(x) 
        # x = self.activation(x)
        x = self.maxpool(x)

        for stage in self.stages:
            x = stage(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1) 
        x = self.fc(x)

        return x

def create_resnet18(num_classes: int = 1000, config: ResNetConfig = original_config) -> ResNet:
    """Creates a ResNet model with the specified configuration."""
    return ResNet(config=config, num_classes=num_classes)