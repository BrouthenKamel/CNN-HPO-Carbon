import torch
import torch.nn as nn
from typing import Optional
from src.loading.models.simple_CNN.hp import SimpleCNNHP
from src.loading.models.simple_CNN.config import SimpleCNNConfig
from src.schema.model import ModelArchitecture
from src.schema.block import CNNBlock, MLPBlock
from src.schema.layer import (
    ActivationType,
    ConvLayer,
    PoolingLayer,
    DropoutLayer,
    LinearLayer,
    ActivationLayer,
    BatchNormLayer,
    AdaptivePoolingLayer
)
from src.loading.models.simple_CNN.hp import *

_activation_factory = {
    ActivationType.RELU: lambda: nn.ReLU(inplace=True),
    ActivationType.SIGMOID: nn.Sigmoid,
    ActivationType.TANH: nn.Tanh,
    ActivationType.SOFTMAX: lambda: nn.Softmax(dim=1),
}

class SimpleCNN(nn.Module):
    """
    A simple convolutional neural network built from a ModelArchitecture.
    """
    def __init__(self, config: ModelArchitecture, num_classes: int):
        super().__init__()
        layers = []

        # Initial convolution layer
        base_filters = None
        if config.initial_conv_layer is not None:
            ic = config.initial_conv_layer
            base_filters = ic.filters
            layers.append(
                nn.Conv2d(3, ic.filters, kernel_size=ic.kernel_size, stride=ic.stride, padding=ic.padding)
            )
            layers.append(nn.BatchNorm2d(ic.filters))

        # CNN blocks
        for block in config.cnn_blocks:
            conv = block.conv_layer
            in_ch = base_filters or conv.filters
            layers.append(
                nn.Conv2d(
                    in_ch,
                    conv.filters,
                    kernel_size=conv.kernel_size,
                    stride=conv.stride,
                    padding=conv.padding
                )
            )
            if block.batch_norm_layer:
                layers.append(nn.BatchNorm2d(conv.filters))
            if block.activation_layer:
                act_type = block.activation_layer.type
                layers.append(_activation_factory[act_type]() if callable(_activation_factory[act_type]) else _activation_factory[act_type]())
            if block.pooling_layer:
                p = block.pooling_layer
                layers.append(
                    nn.MaxPool2d(kernel_size=p.kernel_size, stride=p.stride, padding=p.padding)
                )
            base_filters = conv.filters

        # Adaptive or default pooling
        out_size = config.adaptive_pooling_layer.output_size if config.adaptive_pooling_layer else 1
        layers.append(nn.AdaptiveAvgPool2d(out_size))

        self.features = nn.Sequential(*layers)

        # Compute flatten size
        flatten_size = base_filters * (out_size ** 2)

        # Classifier MLP
        mlp_layers = []
        in_features = flatten_size
        for mlp in config.mlp_blocks:
            mlp_layers.append(nn.Linear(in_features, mlp.linear_layer.neurons))
            if mlp.activation_layer:
                act_type = mlp.activation_layer.type
                mlp_layers.append(_activation_factory[act_type]() if callable(_activation_factory[act_type]) else _activation_factory[act_type]())
            if mlp.dropout_layer:
                mlp_layers.append(nn.Dropout(p=mlp.dropout_layer.rate))
            in_features = mlp.linear_layer.neurons

        # Final classification layer
        mlp_layers.append(nn.Linear(in_features, num_classes))
        self.classifier = nn.Sequential(*mlp_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


if __name__ == "__main__":
    from src.loading.models.simple_CNN.hp import ConvLayerHP, CNNBlockHP, MLPBlockHP, SimpleCNNHP
    # Create a more complex HP setup
    hp = SimpleCNNHP(
        initial_conv=ConvLayerHP(filters=32, kernel_size=3, stride=1, padding=1),
        cnn_block_hps=[
            CNNBlockHP(conv=ConvLayerHP(64, 3, 1, 1), batch_norm=True, activation="RELU", pooling={"type": "MAX", "kernel_size": 2, "stride": 2, "padding": 0}),
            CNNBlockHP(conv=ConvLayerHP(128, 3, 1, 1), batch_norm=True, activation="RELU", pooling={"type": "MAX", "kernel_size": 2, "stride": 2, "padding": 0}),
        ],
        adaptive_pooling=4,
        mlp_block_hps=[
            MLPBlockHP(neurons=256, activation="RELU", dropout=0.5),
            MLPBlockHP(neurons=128, activation="RELU", dropout=0.2),
        ],
        training={"epochs": 20, "batch_size": 64, "learning_rate": 1e-3, "optimizer": "ADAM", "momentum": 0.9}
    )
    config = SimpleCNNConfig.from_hp(hp)
    model = SimpleCNN(config=config, num_classes=10)
    print(model)

    # Test forward pass
    dummy_input = torch.randn(8, 3, 64, 64)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")

