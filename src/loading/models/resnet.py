import torch
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
if src_dir not in sys.path:
    sys.path.append(src_dir)

from schema.model import ModelArchitecture
from schema.block import ResNetBlock
from schema.layer import PoolingType, ActivationType
from schema.training import OptimizerType


ResNet18Architecture = ModelArchitecture(
    initial_conv_layer=dict(filters=64, kernel_size=7, stride=2, padding=3),
    initial_bn_layer=dict(), 
    initial_activation_layer=dict(type=ActivationType.RELU.value),
    initial_pooling_layer=dict(type=PoolingType.MAX.value, kernel_size=3, stride=2, padding=1),

    resnet_blocks=[
        ResNetBlock(in_channels=64, out_channels=64, stride=1, activation_type=ActivationType.RELU.value),
        ResNetBlock(in_channels=64, out_channels=64, stride=1, activation_type=ActivationType.RELU.value),
        ResNetBlock(
            in_channels=64, 
            out_channels=128, 
            stride=2, 
            activation_type=ActivationType.RELU.value,
            downsample=dict(filters=128, kernel_size=1, stride=2, padding=0)  
        ),
        ResNetBlock(in_channels=128, out_channels=128, stride=1, activation_type=ActivationType.RELU.value),
        ResNetBlock(
            in_channels=128, 
            out_channels=256, 
            stride=2, 
            activation_type=ActivationType.RELU.value,
            downsample=dict(filters=256, kernel_size=1, stride=2, padding=0)  
        ),
        ResNetBlock(in_channels=256, out_channels=256, stride=1, activation_type=ActivationType.RELU.value),
        ResNetBlock(
            in_channels=256, 
            out_channels=512, 
            stride=2, 
            activation_type=ActivationType.RELU.value,
            downsample=dict(filters=512, kernel_size=1, stride=2, padding=0) 
        ),
        ResNetBlock(in_channels=512, out_channels=512, stride=1, activation_type=ActivationType.RELU.value),
    ],

    cnn_blocks=[],

    adaptive_pooling_layer=dict(
        type=PoolingType.AVG.value,
        output_size=1  
    ),
    mlp_blocks=[], 
    training=dict(
        epochs=20,
        batch_size=64,
        learning_rate=0.01,
        optimizer=OptimizerType.SGD.value,
        momentum=0.9,
        weight_decay=0.0001
    )
)

if __name__ == '__main__':
    try:
        from loading.models.model_builder import create_model
        
        model = create_model(ResNet18Architecture, in_channels=3, num_classes=10)
        print("ResNet18 Schema Model (structure based on schema):")
        print(model) 

        dummy_input = torch.randn(2, 3, 224, 224) 
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        print("Lazy layers initialized.")
        print("Output shape:", output.shape)
        model.train()
    except ImportError as e:
        print(f"Import error: {e}")
        print("Try running this file from the project root with:")
        print("python -m src.loading.models.resnet")
    except Exception as e:
        print(f"Error during model creation or dummy forward pass: {e}")
        print("(This might be expected if model_builder.py is not yet updated for ResNetBlock)")

