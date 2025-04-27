from typing import Union, Type

import torch.nn as nn

from src.schema.model import ModelArchitecture
from src.schema.block import ResNetBlock
from src.schema.layer import PoolingType, ActivationType, PaddingType

def _get_activation(activation_type_str: str) -> nn.Module:
    activation_type = ActivationType(activation_type_str)
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

def _get_padding(padding: Union[int, str]) -> Union[int, str]:
    if isinstance(padding, str):
        try:
            padding_type = PaddingType(padding)
            if padding_type == PaddingType.SAME:
                return 'same'
            elif padding_type == PaddingType.VALID:
                return 0
            else:
                 raise ValueError(f"Unsupported padding type string: {padding}")
        except ValueError:
             raise ValueError(f"Invalid string value for padding: {padding}")
    elif isinstance(padding, int):
        return padding
    else:
        raise ValueError(f"Unsupported padding type: {type(padding)}")

class ResNetBlockModule(nn.Module):
    expansion = 1

    def __init__(self, schema_block: ResNetBlock, activation_fn: Type[nn.Module]):
        super().__init__()
        self.in_channels = schema_block.in_channels
        self.out_channels = schema_block.out_channels
        self.stride = schema_block.stride
        self.activation = activation_fn()

        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=self.stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_channels)

        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels * self.expansion, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.out_channels * self.expansion)

        self.downsample = None
        if schema_block.downsample:
            ds_padding = _get_padding(schema_block.downsample.padding)
            self.downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels * self.expansion,
                          kernel_size=schema_block.downsample.kernel_size,
                          stride=schema_block.downsample.stride,
                          padding=ds_padding,
                          bias=False),
                nn.BatchNorm2d(self.out_channels * self.expansion)
            )
        elif self.stride != 1 or self.in_channels != self.out_channels * self.expansion:
             print(f"Warning: Creating default 1x1 conv downsample for ResNet block {self.in_channels}->{self.out_channels} stride {self.stride}")
             self.downsample = nn.Sequential(
                 nn.Conv2d(self.in_channels, self.out_channels * self.expansion, kernel_size=1, stride=self.stride, bias=False),
                 nn.BatchNorm2d(self.out_channels * self.expansion),
             )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activation(out)

        return out

def create_model(model_architecture: ModelArchitecture, in_channels: int, num_classes: int) -> nn.Module:
    """
    Builds a PyTorch nn.Module from a ModelArchitecture schema, now supporting ResNetBlocks.
    """
    class Model(nn.Module):
        
        def __init__(self):
            super().__init__()
            
            self.feature_extractor_layers = nn.ModuleList()
            current_channels = in_channels
            initial_layers_defined = False
            
            if model_architecture.initial_conv_layer:
                initial_layers_defined = True
                conv_schema = model_architecture.initial_conv_layer
                padding_val = _get_padding(conv_schema.padding)
                self.feature_extractor_layers.append(nn.Conv2d(
                    current_channels, conv_schema.filters, conv_schema.kernel_size,
                    conv_schema.stride, padding_val, bias=False 
                ))
                current_channels = conv_schema.filters
                if model_architecture.initial_bn_layer:
                    self.feature_extractor_layers.append(nn.BatchNorm2d(current_channels))
                if model_architecture.initial_activation_layer:
                    activation_type = model_architecture.initial_activation_layer.get('type') if isinstance(model_architecture.initial_activation_layer, dict) else model_architecture.initial_activation_layer.type
                    self.feature_extractor_layers.append(_get_activation(activation_type))
                if model_architecture.initial_pooling_layer:
                    pool_schema = model_architecture.initial_pooling_layer
                    pool_padding_val = _get_padding(pool_schema.padding)
                    pool_type = pool_schema.get('type') if isinstance(pool_schema, dict) else pool_schema.type
                    if pool_type == PoolingType.MAX.value:
                        self.feature_extractor_layers.append(nn.MaxPool2d(pool_schema.kernel_size, pool_schema.stride, pool_padding_val))
                    elif pool_type == PoolingType.AVG.value:
                        self.feature_extractor_layers.append(nn.AvgPool2d(pool_schema.kernel_size, pool_schema.stride, pool_padding_val))

            if model_architecture.resnet_blocks:
                if not initial_layers_defined:
                     print("Warning: ResNet blocks defined but no initial layers. Ensure current_channels is correct.")
                first_block = model_architecture.resnet_blocks[0]
                activation_type = first_block.get('activation_type') if isinstance(first_block, dict) else first_block.activation_type
                activation_fn_type = _get_activation(activation_type).__class__
                for block_schema in model_architecture.resnet_blocks:
                    if block_schema.in_channels != current_channels:
                         print(f"Warning: ResNetBlock in_channels ({block_schema.in_channels}) doesn't match expected ({current_channels}). Using schema value.")

                    self.feature_extractor_layers.append(ResNetBlockModule(block_schema, activation_fn_type))
                    current_channels = block_schema.out_channels * ResNetBlockModule.expansion 

            elif model_architecture.cnn_blocks:
                if initial_layers_defined:
                     print("Warning: Initial layers defined but using standard CNN blocks. Ensure channel counts align.")
                for block in model_architecture.cnn_blocks:
                    padding_val = _get_padding(block.conv_layer.padding)
                    conv = nn.Conv2d(
                        in_channels=current_channels,
                        out_channels=block.conv_layer.filters,
                        kernel_size=block.conv_layer.kernel_size,
                        stride=block.conv_layer.stride,
                        padding=padding_val,
                        bias=not bool(block.batch_norm_layer)
                    )
                    self.feature_extractor_layers.append(conv)
                    current_channels = block.conv_layer.filters

                    if block.batch_norm_layer:
                        self.feature_extractor_layers.append(nn.BatchNorm2d(current_channels))

                    if block.activation_layer:
                        self.feature_extractor_layers.append(_get_activation(block.activation_layer.type))

                    if block.pooling_layer:
                        pool_padding_val = _get_padding(block.pooling_layer.padding)
                        if block.pooling_layer.type == PoolingType.MAX.value:
                            self.feature_extractor_layers.append(nn.MaxPool2d(
                                kernel_size=block.pooling_layer.kernel_size,
                                stride=block.pooling_layer.stride,
                                padding=pool_padding_val
                            ))
                        elif block.pooling_layer.type == PoolingType.AVG.value:
                            self.feature_extractor_layers.append(nn.AvgPool2d(
                                kernel_size=block.pooling_layer.kernel_size,
                                stride=block.pooling_layer.stride,
                                padding=pool_padding_val
                            ))

            if model_architecture.adaptive_pooling_layer:
                if model_architecture.adaptive_pooling_layer.type == PoolingType.AVG.value:
                    self.adaptive_pool = nn.AdaptiveAvgPool2d(
                        (model_architecture.adaptive_pooling_layer.output_size,
                         model_architecture.adaptive_pooling_layer.output_size)
                    )
                elif model_architecture.adaptive_pooling_layer.type == PoolingType.MAX.value:
                    self.adaptive_pool = nn.AdaptiveMaxPool2d(
                        (model_architecture.adaptive_pooling_layer.output_size,
                         model_architecture.adaptive_pooling_layer.output_size)
                    )
                else:
                    raise ValueError("Unsupported adaptive pooling type")
            else:
                self.adaptive_pool = None

            self.flatten = nn.Flatten()

            mlp_layers = []
            first_linear = True
            prev_neurons = None
            for block in model_architecture.mlp_blocks:
                if block.dropout_layer:
                    mlp_layers.append(nn.Dropout(p=block.dropout_layer.rate))
                if block.linear_layer:
                    if first_linear:
                        mlp_layers.append(nn.LazyLinear(block.linear_layer.neurons))
                        prev_neurons = block.linear_layer.neurons
                        first_linear = False
                    else:
                        if prev_neurons is None:
                             raise ValueError("Cannot determine input features for a non-first Linear layer in MLP.")
                        mlp_layers.append(nn.Linear(prev_neurons, block.linear_layer.neurons))
                        prev_neurons = block.linear_layer.neurons
                if block.activation_layer:
                    mlp_layers.append(_get_activation(block.activation_layer.type))

            if first_linear:
                mlp_layers.append(nn.LazyLinear(num_classes))
            else:
                 if prev_neurons is None:
                     raise ValueError("Cannot determine input features for the final classification layer.")
                 mlp_layers.append(nn.Linear(prev_neurons, num_classes))

            self.classifier = nn.Sequential(*mlp_layers)

        def forward(self, x):
            for layer in self.feature_extractor_layers:
                x = layer(x)

            if self.adaptive_pool:
                x = self.adaptive_pool(x)

            x = self.flatten(x)

            x = self.classifier(x)
            return x

    model = Model()
    
    return model
