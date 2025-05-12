import torch.nn as nn
from typing import List, Optional, Type, Callable, Union

from src.loading.models.resnet18.hp import (
    ConvLayerHP,
    BasicBlockHP,
    ClassifierHP,
    ResNetHP,
    original_hp
)

def get_activation_layer(activation_name: str) -> Type[nn.Module]:
    if activation_name == "ReLU":
        return nn.ReLU
    elif activation_name == "Hardswish":
        return nn.Hardswish
    elif activation_name == "LeakyReLU":
        return nn.LeakyReLU

class ConvLayerConfig:
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 activation_layer: Type[nn.Module],
                 padding: Optional[int] = None, 
                 bias: bool = False): 
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding if padding is not None else (kernel_size - 1) // 2
        self.activation_layer = activation_layer
        self.bias = bias
        self.norm_layer: Type[nn.BatchNorm2d] = nn.BatchNorm2d
        self.norm_layer_args: dict = {'num_features': out_channels}

    @staticmethod
    def from_hp(hp: ConvLayerHP, in_channels: int) -> 'ConvLayerConfig':
        return ConvLayerConfig(
            in_channels=in_channels,
            out_channels=hp.filters,
            kernel_size=hp.kernel_size,
            stride=hp.stride,
            activation_layer=get_activation_layer(hp.activation)
        )

class BasicBlockConfig:
    def __init__(self,
                 conv1_config: ConvLayerConfig,
                 conv2_config: ConvLayerConfig,
                 activation_layer: Type[nn.Module],
                 stride: int, 
                 downsample_config: Optional[ConvLayerConfig] = None): 
        self.conv1_config = conv1_config
        self.conv2_config = conv2_config
        self.activation_layer = activation_layer
        self.stride = stride
        self.downsample_config = downsample_config

class ClassifierConfig:
    def __init__(self,
                 in_features: int,
                 neurons: List[int],
                 activation_layer: Type[nn.Module],
                 dropout_rate: float):
        self.in_features = in_features
        self.neurons = neurons
        self.activation_layer = activation_layer
        self.dropout_rate = dropout_rate

    @staticmethod
    def from_hp(hp: ClassifierHP, in_features: int) -> 'ClassifierConfig':
        return ClassifierConfig(
            in_features=in_features,
            neurons=hp.neurons,
            activation_layer=get_activation_layer(hp.activation),
            dropout_rate=hp.dropout_rate
        )

class ResNetConfig:
    def __init__(self,
                 initial_conv_config: ConvLayerConfig,
                 initial_activation_layer: Type[nn.Module], 
                 initial_batch_norm_layer: Type[nn.BatchNorm2d],
                 initial_batch_norm_args: dict,
                 initial_maxpool: bool, 
                 stage_configs: List[List[BasicBlockConfig]], 
                 classifier_config: ClassifierConfig,
                 adaptive_pool_output_size: Union[int, tuple] = (1, 1)):
        self.initial_conv_config = initial_conv_config
        self.initial_activation_layer = initial_activation_layer
        self.initial_batch_norm_layer = initial_batch_norm_layer
        self.initial_batch_norm_args = initial_batch_norm_args
        self.initial_maxpool = initial_maxpool
        self.stage_configs = stage_configs
        self.adaptive_pool_output_size = adaptive_pool_output_size
        self.classifier_config = classifier_config

    @staticmethod
    def from_hp(hp: ResNetHP, input_channels: int = 3) -> 'ResNetConfig':

        initial_conv_config = ConvLayerConfig.from_hp(hp.initial_conv_hp, in_channels=input_channels)
        initial_activation_layer = get_activation_layer(hp.initial_conv_hp.activation)
        initial_batch_norm_layer = nn.BatchNorm2d
        initial_batch_norm_args = {'num_features': initial_conv_config.out_channels}
        current_channels = initial_conv_config.out_channels

        stage_configs: List[List[BasicBlockConfig]] = []
        previous_stage_out_channels = current_channels

        for i, (num_blocks, stage_hp) in enumerate(zip(hp.stage_block_counts, hp.stage_block_hps)):
            stage_list: List[BasicBlockConfig] = []
            stage_activation = get_activation_layer(stage_hp.activation)
            stage_out_channels = stage_hp.filters

            for block_idx in range(num_blocks):
                is_first_block = (block_idx == 0)
                stride = stage_hp.stride if is_first_block else 1
                block_in_channels = previous_stage_out_channels if is_first_block else stage_out_channels

                downsample_config: Optional[ConvLayerConfig] = None
                if stride != 1 or block_in_channels != stage_out_channels:
                    downsample_config = ConvLayerConfig(
                        in_channels=block_in_channels,
                        out_channels=stage_out_channels,
                        kernel_size=1, 
                        stride=stride, 
                        activation_layer=nn.Identity, 
                        padding=0,
                        bias=False
                    )

                conv1_config = ConvLayerConfig(
                    in_channels=block_in_channels,
                    out_channels=stage_out_channels,
                    kernel_size=3, 
                    stride=stride,
                    activation_layer=stage_activation, 
                    padding=1, 
                    bias=False
                )
                conv2_config = ConvLayerConfig(
                    in_channels=stage_out_channels,
                    out_channels=stage_out_channels,
                    kernel_size=3, 
                    stride=1,
                    activation_layer=nn.Identity, # Activation happens *after* skip connection
                    padding=1,
                    bias=False
                )

                block_config = BasicBlockConfig(
                    conv1_config=conv1_config,
                    conv2_config=conv2_config,
                    activation_layer=stage_activation, 
                    stride=stride,
                    downsample_config=downsample_config
                )
                stage_list.append(block_config)

            stage_configs.append(stage_list)
            previous_stage_out_channels = stage_out_channels

        classifier_in_features = previous_stage_out_channels
        classifier_config = ClassifierConfig.from_hp(hp.classifier_hp, in_features=classifier_in_features)

        return ResNetConfig(
            initial_conv_config=initial_conv_config,
            initial_activation_layer=initial_activation_layer,
            initial_batch_norm_layer=initial_batch_norm_layer,
            initial_batch_norm_args=initial_batch_norm_args,
            initial_maxpool=hp.initial_maxpool,
            stage_configs=stage_configs,
            adaptive_pool_output_size=(1, 1), 
            classifier_config=classifier_config
        )

original_config = ResNetConfig.from_hp(original_hp)
