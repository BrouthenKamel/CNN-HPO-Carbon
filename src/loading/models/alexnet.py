import torch
import torch.nn as nn

from src.schema.model import ModelArchitecture
from src.schema.block import CNNBlock, MLPBlock
from src.schema.layer import ConvLayer, PoolingLayer, DropoutLayer, LinearLayer, ActivationLayer, BatchNormLayer, AdaptivePoolingLayer
from src.schema.training import TrainingParams
from src.schema.layer import PoolingType, ActivationType
from src.schema.training import OptimizerType

# AlexNet architecture representation - Updated for Pydantic compatibility
AlexNetArchitecture = ModelArchitecture(
    cnn_blocks=[
        CNNBlock(
            conv_layer=ConvLayer(filters=96, kernel_size=5, stride=1, padding=2),
            activation_layer=ActivationLayer(type=ActivationType.RELU.value),
            pooling_layer=PoolingLayer(type=PoolingType.MAX.value, kernel_size=2, stride=2, padding=0),
            batch_norm_layer=None
        ),
        CNNBlock(
            conv_layer=ConvLayer(filters=192, kernel_size=5, stride=1, padding=2),
            activation_layer=ActivationLayer(type=ActivationType.RELU.value),
            pooling_layer=PoolingLayer(type=PoolingType.MAX.value, kernel_size=2, stride=2, padding=0),
            batch_norm_layer=None
        ),
        CNNBlock(
            conv_layer=ConvLayer(filters=384, kernel_size=3, stride=1, padding=1),
            activation_layer=ActivationLayer(type=ActivationType.RELU.value),
            pooling_layer=None,
            batch_norm_layer=None
        ),
        CNNBlock(
            conv_layer=ConvLayer(filters=256, kernel_size=3, stride=1, padding=1),
            activation_layer=ActivationLayer(type=ActivationType.RELU.value),
            pooling_layer=None,
            batch_norm_layer=None
        ),
        CNNBlock(
            conv_layer=ConvLayer(filters=256, kernel_size=3, stride=1, padding=1),
            activation_layer=ActivationLayer(type=ActivationType.RELU.value),
            pooling_layer=PoolingLayer(type=PoolingType.MAX.value, kernel_size=2, stride=2, padding=0),
            batch_norm_layer=None
        ),
    ],
    adaptive_pooling_layer=AdaptivePoolingLayer(
        type=PoolingType.AVG.value,
        output_size=3
    ),
    mlp_blocks=[
        MLPBlock(
            dropout_layer=DropoutLayer(rate=0.5),
            linear_layer=LinearLayer(neurons=4096),
            activation_layer=ActivationLayer(type=ActivationType.RELU.value),
        ),
        MLPBlock(
            dropout_layer=DropoutLayer(rate=0.5),
            linear_layer=LinearLayer(neurons=4096),
            activation_layer=ActivationLayer(type=ActivationType.RELU.value),
        )
    ],
    training_params=TrainingParams(
        epochs=10,
        batch_size=32,
        learning_rate=0.001,
        optimizer=OptimizerType.SGD.value,
        momentum=None,
        weight_decay=None
    )
)

class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
