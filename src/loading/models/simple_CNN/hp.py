from src.schema.model import ModelArchitecture
from src.schema.block import CNNBlock, MLPBlock
from src.schema.layer import ConvLayer, PoolingLayer, DropoutLayer, LinearLayer, ActivationLayer, BatchNormLayer, AdaptivePoolingLayer
from src.schema.training import TrainingParams
from src.schema.layer import PoolingType, ActivationType
from src.schema.training import OptimizerType



SimpleCNNArchitecture = ModelArchitecture(
    initial_conv_layer=ConvLayer(filters=16, kernel_size=3, stride=1, padding=1),

    cnn_blocks=[
        CNNBlock(
            conv_layer=ConvLayer(filters=32, kernel_size=3, stride=1, padding=0),
            activation_layer=ActivationLayer(type=ActivationType.RELU.value),
            pooling_layer=PoolingLayer(type=PoolingType.MAX.value, kernel_size=2, stride=1, padding=0),
            batch_norm_layer=None
        ),
        CNNBlock(
            conv_layer=ConvLayer(filters=64, kernel_size=3, stride=1, padding=0),
            activation_layer=None,
            pooling_layer=PoolingLayer(type=PoolingType.MAX.value, kernel_size=2, stride=1, padding=0),
            batch_norm_layer=None
        ),
    ],
    adaptive_pooling_layer=None,
    mlp_blocks=[
        MLPBlock(
            dropout_layer=None,
            linear_layer=LinearLayer(neurons=128),
            activation_layer=ActivationLayer(type=ActivationType.RELU.value),
        ),
        MLPBlock(
            dropout_layer=DropoutLayer(rate=0.2),
            linear_layer=LinearLayer(neurons=64),
            activation_layer=ActivationLayer(type=ActivationType.RELU.value),
        )
    ],
    training_params=TrainingParams(
        epochs=20,
        batch_size=32,
        learning_rate=0.01,
        optimizer=OptimizerType.ADAM.value,
        momentum=None,
        weight_decay=None
    )
)
