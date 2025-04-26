import torch
import torch.nn as nn
from typing import Union

from src.schema.model import ModelArchitecture
from src.schema.block import CNNBlock, MLPBlock
from src.schema.layer import ConvLayer, PoolingLayer, PoolingType, DropoutLayer, LinearLayer, ActivationLayer, ActivationType, PaddingType, AdaptivePoolingLayer
from src.schema.training import Training, OptimizerType

def create_model(model_architecture: ModelArchitecture, in_channels: int, num_classes: int) -> nn.Module:
    
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []

            current_in_channels = in_channels

            # Build CNN blocks
            for block in model_architecture.cnn_blocks:
                # Conv layer
                conv = nn.Conv2d(
                    in_channels=current_in_channels,
                    out_channels=block.conv_layer.filters,
                    kernel_size=block.conv_layer.kernel_size,
                    stride=block.conv_layer.stride,
                    padding=self._get_padding(block.conv_layer.padding)
                )
                layers.append(conv)
                current_in_channels = block.conv_layer.filters

                # Activation layer
                if block.activation_layer:
                    layers.append(self._get_activation(block.activation_layer.type))

                # Pooling layer
                if block.pooling_layer:
                    if block.pooling_layer.type == PoolingType.MAX:
                        layers.append(nn.MaxPool2d(
                            kernel_size=block.pooling_layer.kernel_size,
                            stride=block.pooling_layer.stride,
                            padding=self._get_padding(block.pooling_layer.padding)
                        ))
                    elif block.pooling_layer.type == PoolingType.AVG:
                        layers.append(nn.AvgPool2d(
                            kernel_size=block.pooling_layer.kernel_size,
                            stride=block.pooling_layer.stride,
                            padding=self._get_padding(block.pooling_layer.padding)
                        ))

            self.features = nn.Sequential(*layers)

            # Adaptive pooling
            if model_architecture.adaptive_pooling_layer:
                if model_architecture.adaptive_pooling_layer.type == PoolingType.AVG:
                    self.adaptive_pool = nn.AdaptiveAvgPool2d(
                        (model_architecture.adaptive_pooling_layer.output_size,
                         model_architecture.adaptive_pooling_layer.output_size)
                    )
                elif model_architecture.adaptive_pooling_layer.type == PoolingType.MAX:
                    self.adaptive_pool = nn.AdaptiveMaxPool2d(
                        (model_architecture.adaptive_pooling_layer.output_size,
                         model_architecture.adaptive_pooling_layer.output_size)
                    )
                else:
                    raise ValueError("Unsupported adaptive pooling type")
            else:
                self.adaptive_pool = None

            # Flatten before MLP
            self.flatten = nn.Flatten()

            # Build MLP blocks
            mlp_layers = []
            first_linear = True

            for block in model_architecture.mlp_blocks:
                if block.dropout_layer:
                    mlp_layers.append(nn.Dropout(p=block.dropout_layer.rate))

                if block.linear_layer:
                    if first_linear:
                        # First linear input is unknown size, use LazyLinear
                        mlp_layers.append(nn.LazyLinear(block.linear_layer.neurons))
                        prev_neurons = block.linear_layer.neurons
                        first_linear = False
                    else:
                        mlp_layers.append(nn.Linear(prev_neurons, block.linear_layer.neurons))
                        prev_neurons = block.linear_layer.neurons

                if block.activation_layer:
                    mlp_layers.append(self._get_activation(block.activation_layer.type))

            # Final output layer
            mlp_layers.append(nn.Linear(prev_neurons, num_classes))

            self.classifier = nn.Sequential(*mlp_layers)

        def forward(self, x):
            x = self.features(x)
            if self.adaptive_pool:
                x = self.adaptive_pool(x)
            x = self.flatten(x)
            x = self.classifier(x)
            return x

        def _get_activation(self, activation_type: ActivationType) -> nn.Module:
            if activation_type == ActivationType.RELU:
                return nn.ReLU(inplace=True)
            elif activation_type == ActivationType.SIGMOID:
                return nn.Sigmoid()
            elif activation_type == ActivationType.TANH:
                return nn.Tanh()
            elif activation_type == ActivationType.SOFTMAX:
                return nn.Softmax(dim=1)
            else:
                raise ValueError(f"Unsupported activation type: {activation_type}")

        def _get_padding(self, padding: Union[int, PaddingType]) -> Union[int, str]:
            if isinstance(padding, int):
                return padding
            if padding == PaddingType.SAME:
                return 'same'
            elif padding == PaddingType.VALID:
                return 0
            else:
                raise ValueError(f"Unsupported padding type: {padding}")

    return Model()

# AlexNet architecture representation
AlexNetArchitecture = ModelArchitecture(
    cnn_blocks=[
        CNNBlock(
            conv_layer=ConvLayer(filters=64, kernel_size=5, stride=1, padding=2),
            activation_layer=ActivationLayer(type=ActivationType.RELU.value),
            pooling_layer=PoolingLayer(type=PoolingType.MAX.value, kernel_size=2, stride=2, padding=0),
        ),
        CNNBlock(
            conv_layer=ConvLayer(filters=192, kernel_size=5, stride=1, padding=2),
            activation_layer=ActivationLayer(type=ActivationType.RELU.value),
            pooling_layer=PoolingLayer(type=PoolingType.MAX.value, kernel_size=2, stride=2, padding=0),
        ),
        CNNBlock(
            conv_layer=ConvLayer(filters=384, kernel_size=3, stride=1, padding=1),
            activation_layer=ActivationLayer(type=ActivationType.RELU.value),
        ),
        CNNBlock(
            conv_layer=ConvLayer(filters=256, kernel_size=3, stride=1, padding=1),
            activation_layer=ActivationLayer(type=ActivationType.RELU.value),
        ),
        CNNBlock(
            conv_layer=ConvLayer(filters=256, kernel_size=3, stride=1, padding=1),
            activation_layer=ActivationLayer(type=ActivationType.RELU.value),
            pooling_layer=PoolingLayer(type=PoolingType.MAX.value, kernel_size=2, stride=2, padding=0)
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
            linear_layer=LinearLayer(neurons=(4096)),
            activation_layer=ActivationLayer(type=ActivationType.RELU.value),
        )
    ],
    training=Training(
        epochs=10,
        batch_size=32,
        learning_rate=0.001,
        optimizer=OptimizerType.SGD.value,
        momentum=None,
        weight_decay=None
    )
)

# for reference, the original AlexNet architecture
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
    
    model = create_model(AlexNetArchitecture, in_channels=3, num_classes=10)
    alexnet = AlexNet(num_classes=10)
    
    x = torch.randn(1, 3, 224, 224)
    
    output = model(x)
    print(output.shape)
    print(output)
    
    output = alexnet(x)
    print(output.shape)
    print(output)
