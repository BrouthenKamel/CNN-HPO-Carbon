import torch.nn as nn

from functools import partial

from src.loading.models.mobilenet.utils import make_divisible
from src.loading.models.mobilenet.hp import MobileNetHP, original_hp

def get_activation_layer(activation_layer):
    if activation_layer == "ReLU":
        return nn.ReLU
    elif activation_layer == "LeakyReLU":
        return nn.LeakyReLU
    elif activation_layer == "PReLU":
        return nn.PReLU
    elif activation_layer == "ELU":
        return nn.ELU
    elif activation_layer == "Sigmoid":
        return nn.Sigmoid
    elif activation_layer == "Tanh":
        return nn.Tanh
    elif activation_layer == "Hardswish":
        return nn.Hardswish
    else:
        raise ValueError(f"Unsupported activation layer: {activation_layer}")

class SqueezeExcitationConfig:
    def __init__(self, squeeze_factor: int = 4, activation_layer: str = "Hardsigmoid"):
        self.squeeze_factor = squeeze_factor
        self.activation_layer = self._get_activation_layer(activation_layer)
        
    def _get_activation_layer(self, activation_layer):
        if activation_layer == "Hardsigmoid":
            return nn.Hardsigmoid
        elif activation_layer == "Sigmoid":
            return nn.Sigmoid
        else:
            raise ValueError(f"Unsupported activation layer: {self.activation_layer}")
        
class ConvBNActivationConfig:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int,
                 norm_layer: str = "BatchNorm2d", activation_layer: str = "ReLU", eps: float = None, momentum: float = None, ignore_in_channels: bool = False):
        if ignore_in_channels:
            self.in_channels = in_channels
        else:
            self.in_channels = make_divisible(in_channels, 8)
        self.out_channels = make_divisible(out_channels, 8)
        self.kernel_size = kernel_size
        self.stride = stride
        self.norm_layer = self.get_norm_layer(norm_layer, eps, momentum)
        
        self.activation_layer = get_activation_layer(activation_layer)
        
    def get_norm_layer(self, norm_layer, eps, momentum):
        if norm_layer == "BatchNorm2d":
            kwargs = {}
            if eps != None:
                kwargs['eps'] = eps
            if momentum != None:
                kwargs['momentum'] = momentum
            return partial(nn.BatchNorm2d, **kwargs)
        else:
            raise ValueError(f"Unsupported normalization layer: {self.norm_layer}")
        
class InvertedResidualConfig:
    def __init__(self, expand_channels: int, use_se: bool, se_config: SqueezeExcitationConfig, conv_bn_activation_config: ConvBNActivationConfig):
        self.use_se = use_se
        self.se_config = se_config
        self.conv_bn_activation_config = conv_bn_activation_config
        self.expanded_channels = make_divisible(expand_channels, 8)
        
class ClassifierConfig:
    def __init__(self, neurons: int, activation_layer: str, dropout_rate: float):
        
        self.neurons = neurons
        self.activation_layer = get_activation_layer(activation_layer)
        self.dropout_rate = dropout_rate
        
class MobileNetConfig:
    def __init__(self, initial_conv_config, last_conv_upsample, last_conv_config, inverted_residual_configs, classifier_config):
        self.initial_conv_config = initial_conv_config
        self.inverted_residual_configs = inverted_residual_configs
        self.last_conv_upsample = last_conv_upsample
        self.last_conv_config = last_conv_config
        self.classifier_config = classifier_config

    @staticmethod
    def from_hp(hp: MobileNetHP):
        initial_conv_config = ConvBNActivationConfig(
            in_channels=3,
            out_channels=hp.initial_conv_hp.channels,
            kernel_size=hp.initial_conv_hp.kernel_size,
            stride=hp.initial_conv_hp.stride,
            activation_layer=hp.initial_conv_hp.activation,
            norm_layer="BatchNorm2d",
            eps=1e-3,
            momentum=1e-2,
            ignore_in_channels=True
        )
        
        inverted_residual_configs = []
        in_channels = hp.initial_conv_hp.channels
        
        for ir_hp in hp.inverted_residual_hps:
            conv_hp = ir_hp.conv_bn_activation_hp
            se_config = (
                SqueezeExcitationConfig(
                    squeeze_factor=ir_hp.se_hp.squeeze_factor,
                    activation_layer=ir_hp.se_hp.activation
                ) if ir_hp.use_se and ir_hp.se_hp else None
            )
            
            conv_bn_activation_config = ConvBNActivationConfig(
                in_channels=in_channels,
                out_channels=conv_hp.channels,
                kernel_size=conv_hp.kernel_size,
                stride=conv_hp.stride,
                activation_layer=conv_hp.activation,
                norm_layer="BatchNorm2d"
            )
            
            inverted_residual_configs.append(InvertedResidualConfig(
                expand_channels=ir_hp.expanded_channels,
                use_se=ir_hp.use_se,
                se_config=se_config,
                conv_bn_activation_config=conv_bn_activation_config
            ))
            
            in_channels = conv_hp.channels
        
        last_conv_in_channels = in_channels
        last_conv_out_channels = last_conv_in_channels * hp.last_conv_upsample
        
        last_conv_config = ConvBNActivationConfig(
            in_channels=last_conv_in_channels,
            out_channels=last_conv_out_channels,
            kernel_size=hp.last_conv_hp.kernel_size,
            stride=hp.last_conv_hp.stride,
            activation_layer=hp.last_conv_hp.activation,
            norm_layer="BatchNorm2d",
            eps=1e-3,
            momentum=1e-2
        )
        
        classifier_config = ClassifierConfig(
            neurons=hp.classifier_hp.neurons,
            activation_layer=hp.classifier_hp.activation,
            dropout_rate=hp.classifier_hp.dropout_rate
        )
        
        return MobileNetConfig(
            initial_conv_config=initial_conv_config,
            inverted_residual_configs=inverted_residual_configs,
            last_conv_upsample=hp.last_conv_upsample,
            last_conv_config=last_conv_config,
            classifier_config=classifier_config
        )

original_config = MobileNetConfig.from_hp(original_hp)
