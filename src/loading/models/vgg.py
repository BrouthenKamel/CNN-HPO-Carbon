import torch
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
if src_dir not in sys.path:
    sys.path.append(src_dir)

from schema.model import ModelArchitecture
from schema.block import CNNBlock, MLPBlock
from schema.layer import PoolingType, ActivationType
from schema.training import OptimizerType

VGG11Architecture = ModelArchitecture(
    cnn_blocks=[
        CNNBlock(
            conv_layer=dict(filters=64, kernel_size=3, stride=1, padding=1), 
            activation_layer=dict(type=ActivationType.RELU.value),
            pooling_layer=dict(type=PoolingType.MAX.value, kernel_size=2, stride=2, padding=0),
            batch_norm_layer=dict(),
        ),
        CNNBlock(
            conv_layer=dict(filters=128, kernel_size=3, stride=1, padding=1), 
            activation_layer=dict(type=ActivationType.RELU.value),
            batch_norm_layer=dict(),
            pooling_layer=dict(type=PoolingType.MAX.value, kernel_size=2, stride=2, padding=0)
        ),
        CNNBlock(
            conv_layer=dict(filters=256, kernel_size=3, stride=1, padding=1), 
            activation_layer=dict(type=ActivationType.RELU.value),
            batch_norm_layer=dict()
        ),
        CNNBlock(
            conv_layer=dict(filters=256, kernel_size=3, stride=1, padding=1), 
            activation_layer=dict(type=ActivationType.RELU.value),
            batch_norm_layer=dict(),
            pooling_layer=dict(type=PoolingType.MAX.value, kernel_size=2, stride=2, padding=0)
        ),
        CNNBlock(
            conv_layer=dict(filters=512, kernel_size=3, stride=1, padding=1), 
            activation_layer=dict(type=ActivationType.RELU.value),
            batch_norm_layer=dict()
        ),
        CNNBlock(
            conv_layer=dict(filters=512, kernel_size=3, stride=1, padding=1), 
            activation_layer=dict(type=ActivationType.RELU.value),
            batch_norm_layer=dict(),
            pooling_layer=dict(type=PoolingType.MAX.value, kernel_size=2, stride=2, padding=0)
        ),
        CNNBlock(
            conv_layer=dict(filters=512, kernel_size=3, stride=1, padding=1), 
            activation_layer=dict(type=ActivationType.RELU.value),
            batch_norm_layer=dict()
        ),
        CNNBlock(
            conv_layer=dict(filters=512, kernel_size=3, stride=1, padding=1), 
            activation_layer=dict(type=ActivationType.RELU.value),
            batch_norm_layer=dict(),
            pooling_layer=dict(type=PoolingType.MAX.value, kernel_size=2, stride=2, padding=0)
        ),
    ],
    adaptive_pooling_layer=dict(
        type=PoolingType.AVG.value, 
        output_size=7 
    ),
    mlp_blocks=[
        MLPBlock(
            linear_layer=dict(neurons=4096), 
            activation_layer=dict(type=ActivationType.RELU.value),
            dropout_layer=dict(rate=0.5)
        ),
        MLPBlock(
            linear_layer=dict(neurons=4096),
            activation_layer=dict(type=ActivationType.RELU.value),
            dropout_layer=dict(rate=0.5)
        )
    ],
    training=dict(
        epochs=15,
        batch_size=64,
        learning_rate=0.001,
        optimizer=OptimizerType.ADAM.value,
        momentum=None,
        weight_decay=None
    )
)

if __name__ == '__main__':
    from model_builder import create_model
    model = create_model(VGG11Architecture, in_channels=3, num_classes=10)
    print("VGG11 Schema Model (structure based on schema):")
    print(model) 

    dummy_input = torch.randn(2, 3, 224, 224) 
    try:
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        print("Lazy layers initialized.")
        print("Output shape:", output.shape)
        model.train()
    except Exception as e:
        print(f"Error during dummy forward pass: {e}")
