from typing import Union

import torch
import torch.nn as nn
from torchvision import models

from src.loading.models.simple_CNN.hp import SimpleCNNArchitecture




class CNNModel(nn.Module):
    def __init__(self, config):
        super(CNNModel, self).__init__()

        layers = []

        # Initial convolutional part (optional)
        if config.initial_conv_layer:
            layers.append(nn.Conv2d(
                in_channels=3,  # Assuming RGB images; modify as needed
                out_channels=config.initial_conv_layer.filters,
                kernel_size=config.initial_conv_layer.kernel_size,
                stride=config.initial_conv_layer.stride,
                padding=config.initial_conv_layer.padding
            ))
        if config.initial_bn_layer:
            layers.append(nn.BatchNorm2d(config.initial_conv_layer.filters))
        if config.initial_activation_layer:
            layers.append(self._get_activation(config.initial_activation_layer["type"]))
        if config.initial_pooling_layer:
            layers.append(self._get_pooling(config.initial_pooling_layer))

        current_channels=None
        # CNN blocks
        for block in config.cnn_blocks:
            current_channels = current_channels if current_channels else config.initial_conv_layer.filters
            conv = nn.Conv2d(
                in_channels=current_channels,
                out_channels=block.conv_layer.filters,
                kernel_size=block.conv_layer.kernel_size,
                stride=block.conv_layer.stride,
                padding=block.conv_layer.padding
            )
            current_channels = block.conv_layer.filters
            layers.append(conv)
            if block.batch_norm_layer:
                layers.append(nn.BatchNorm2d(block.conv_layer.filters))
            if block.activation_layer:
                layers.append(self._get_activation(block.activation_layer.type))
            if block.pooling_layer:
                layers.append(self._get_pooling(block.pooling_layer))

        self.features = nn.Sequential(*layers)

        # Adaptive pooling (optional)
        self.adaptive_pooling = None
        if config.adaptive_pooling_layer:
            self.adaptive_pooling = nn.AdaptiveAvgPool2d(config.adaptive_pooling_layer.output_size)

        # Classifier
        mlp_layers = []
        input_features = None  # You'll have to compute this manually or flatten dynamically
        for block in config.mlp_blocks:
            if block == config.mlp_blocks[-1]:
                output = 10
            else : 
                output = block.linear_layer.neurons
            mlp_layers.append(nn.LazyLinear(output))  # auto infer input size
            if block.activation_layer:
                mlp_layers.append(self._get_activation(block.activation_layer.type))
            if block.dropout_layer:
                mlp_layers.append(nn.Dropout(block.dropout_layer.rate))
            input_features = block.linear_layer.neurons

        self.classifier = nn.Sequential(*mlp_layers)

    def _get_activation(self, act_type):
        act_type = act_type.lower()
        if act_type == "relu":
            return nn.ReLU()
        elif act_type == "sigmoid":
            return nn.Sigmoid()
        elif act_type == "tanh":
            return nn.Tanh()
        elif act_type == "leaky_relu":
            return nn.LeakyReLU()
        else : #return default
            return nn.ReLU()
    def _get_pooling(self, pool_config):
        pool_type = pool_config.type.lower()
        if pool_type == "max":
            return nn.MaxPool2d(kernel_size=pool_config.kernel_size, stride=pool_config.stride, padding=pool_config.padding)
        elif pool_type == "avg":
            return nn.AvgPool2d(kernel_size=pool_config.kernel_size, stride=pool_config.stride, padding=pool_config.padding)
        raise ValueError(f"Unsupported pooling type: {pool_type}")

    def forward(self, x):
        x = self.features(x)
        if self.adaptive_pooling:
            x = self.adaptive_pooling(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x




if __name__ == "__main__":
    
    custom_model = CNNModel(config=SimpleCNNArchitecture)
    
    
