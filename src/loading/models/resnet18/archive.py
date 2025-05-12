from torch import nn
from src.loading.models.resnet18.config import (
    ConvLayerConfig, BasicBlockConfig, ClassifierConfig, ResNetConfig,
)

resnet18_static_config = ResNetConfig(
    initial_conv_config=ConvLayerConfig(
        in_channels=3,
        out_channels=64,
        kernel_size=7,
        stride=2,
        activation_layer=nn.ReLU, 
        padding=3, 
        bias=False
    ),
    initial_activation_layer=nn.ReLU,
    initial_batch_norm_layer=nn.BatchNorm2d,
    initial_batch_norm_args={'num_features': 64},
    initial_maxpool=True,
    stage_configs=[
        [
            BasicBlockConfig( 
                conv1_config=ConvLayerConfig(in_channels=64, out_channels=64, kernel_size=3, stride=1, activation_layer=nn.ReLU, padding=1, bias=False),
                conv2_config=ConvLayerConfig(in_channels=64, out_channels=64, kernel_size=3, stride=1, activation_layer=nn.Identity, padding=1, bias=False),
                activation_layer=nn.ReLU,
                stride=1,
                downsample_config=None 
            ),
            BasicBlockConfig( 
                conv1_config=ConvLayerConfig(in_channels=64, out_channels=64, kernel_size=3, stride=1, activation_layer=nn.ReLU, padding=1, bias=False),
                conv2_config=ConvLayerConfig(in_channels=64, out_channels=64, kernel_size=3, stride=1, activation_layer=nn.Identity, padding=1, bias=False),
                activation_layer=nn.ReLU,
                stride=1,
                downsample_config=None
            ),
        ],

        [
            BasicBlockConfig( 
                conv1_config=ConvLayerConfig(in_channels=64, out_channels=128, kernel_size=3, stride=2, activation_layer=nn.ReLU, padding=1, bias=False),
                conv2_config=ConvLayerConfig(in_channels=128, out_channels=128, kernel_size=3, stride=1, activation_layer=nn.Identity, padding=1, bias=False),
                activation_layer=nn.ReLU,
                stride=2,
                downsample_config=ConvLayerConfig(in_channels=64, out_channels=128, kernel_size=1, stride=2, activation_layer=nn.Identity, padding=0, bias=False) 
            ),
            BasicBlockConfig( 
                conv1_config=ConvLayerConfig(in_channels=128, out_channels=128, kernel_size=3, stride=1, activation_layer=nn.ReLU, padding=1, bias=False),
                conv2_config=ConvLayerConfig(in_channels=128, out_channels=128, kernel_size=3, stride=1, activation_layer=nn.Identity, padding=1, bias=False),
                activation_layer=nn.ReLU,
                stride=1,
                downsample_config=None
            ),
        ],
        [
            BasicBlockConfig( 
                conv1_config=ConvLayerConfig(in_channels=128, out_channels=256, kernel_size=3, stride=2, activation_layer=nn.ReLU, padding=1, bias=False),
                conv2_config=ConvLayerConfig(in_channels=256, out_channels=256, kernel_size=3, stride=1, activation_layer=nn.Identity, padding=1, bias=False),
                activation_layer=nn.ReLU,
                stride=2,
                downsample_config=ConvLayerConfig(in_channels=128, out_channels=256, kernel_size=1, stride=2, activation_layer=nn.Identity, padding=0, bias=False) 
            ),
            BasicBlockConfig(
                conv1_config=ConvLayerConfig(in_channels=256, out_channels=256, kernel_size=3, stride=1, activation_layer=nn.ReLU, padding=1, bias=False),
                conv2_config=ConvLayerConfig(in_channels=256, out_channels=256, kernel_size=3, stride=1, activation_layer=nn.Identity, padding=1, bias=False),
                activation_layer=nn.ReLU,
                stride=1,
                downsample_config=None
            ),
        ],
        [
            BasicBlockConfig( 
                conv1_config=ConvLayerConfig(in_channels=256, out_channels=512, kernel_size=3, stride=2, activation_layer=nn.ReLU, padding=1, bias=False),
                conv2_config=ConvLayerConfig(in_channels=512, out_channels=512, kernel_size=3, stride=1, activation_layer=nn.Identity, padding=1, bias=False),
                activation_layer=nn.ReLU,
                stride=2,
                downsample_config=ConvLayerConfig(in_channels=256, out_channels=512, kernel_size=1, stride=2, activation_layer=nn.Identity, padding=0, bias=False) 
            ),
            BasicBlockConfig( 
                conv1_config=ConvLayerConfig(in_channels=512, out_channels=512, kernel_size=3, stride=1, activation_layer=nn.ReLU, padding=1, bias=False),
                conv2_config=ConvLayerConfig(in_channels=512, out_channels=512, kernel_size=3, stride=1, activation_layer=nn.Identity, padding=1, bias=False),
                activation_layer=nn.ReLU,
                stride=1,
                downsample_config=None
            ),
        ],
    ],
    adaptive_pool_output_size=(1, 1),
    classifier_config=ClassifierConfig(
        in_features=512, 
        neurons=[], 
        activation_layer=nn.ReLU,
        dropout_rate=0.0
    )
)