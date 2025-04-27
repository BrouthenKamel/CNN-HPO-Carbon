import torch
import torch.nn as nn

from schema.model import ModelArchitecture
from schema.block import CNNBlock, MLPBlock
from schema.layer import PoolingType, ActivationType
from schema.training import OptimizerType

# AlexNet architecture representation - Updated for Pydantic compatibility
AlexNetArchitecture = ModelArchitecture(
    cnn_blocks=[
        CNNBlock(
            conv_layer=dict(filters=64, kernel_size=5, stride=1, padding=2),
            activation_layer=dict(type=ActivationType.RELU.value),
            pooling_layer=dict(type=PoolingType.MAX.value, kernel_size=2, stride=2, padding=0),
            batch_norm_layer=dict() 
        ),
        CNNBlock(
            conv_layer=dict(filters=192, kernel_size=5, stride=1, padding=2),
            activation_layer=dict(type=ActivationType.RELU.value),
            pooling_layer=dict(type=PoolingType.MAX.value, kernel_size=2, stride=2, padding=0),
            batch_norm_layer=dict()
        ),
        CNNBlock(
            conv_layer=dict(filters=384, kernel_size=3, stride=1, padding=1),
            activation_layer=dict(type=ActivationType.RELU.value),
            batch_norm_layer=dict()
        ),
        CNNBlock(
            conv_layer=dict(filters=256, kernel_size=3, stride=1, padding=1),
            activation_layer=dict(type=ActivationType.RELU.value),
            batch_norm_layer=dict()
        ),
        CNNBlock(
            conv_layer=dict(filters=256, kernel_size=3, stride=1, padding=1),
            activation_layer=dict(type=ActivationType.RELU.value),
            pooling_layer=dict(type=PoolingType.MAX.value, kernel_size=2, stride=2, padding=0),
            batch_norm_layer=dict()
        ),
    ],
    adaptive_pooling_layer=dict(
        type=PoolingType.AVG.value,
        output_size=3
    ),
    mlp_blocks=[
        MLPBlock(
            dropout_layer=dict(rate=0.5),
            linear_layer=dict(neurons=4096),
            activation_layer=dict(type=ActivationType.RELU.value),
        ),
        MLPBlock(
            dropout_layer=dict(rate=0.5),
            linear_layer=dict(neurons=4096),
            activation_layer=dict(type=ActivationType.RELU.value),
        )
    ],
    training=dict(
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
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
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

if __name__ == "__main__":
    from model_builder import create_model

    model = create_model(AlexNetArchitecture, in_channels=3, num_classes=10)
    alexnet = AlexNet(num_classes=10)

    x = torch.randn(1, 3, 224, 224)
    
    output = model(x)
    print(output.shape)
    print(output)
    
    output = alexnet(x)
    print(output.shape)
    print(output)
