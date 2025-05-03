from typing import Optional
from pydantic import BaseModel

from src.schema.layer import ConvLayer, PoolingLayer, DropoutLayer, LinearLayer, ActivationLayer, BatchNormLayer

class CNNBlock(BaseModel):
    conv_layer: Optional[ConvLayer] = None
    activation_layer: Optional[ActivationLayer] = None
    pooling_layer: Optional[PoolingLayer] = None
    batch_norm_layer: Optional[BatchNormLayer] = None

class MLPBlock(BaseModel):
    dropout_layer: Optional[DropoutLayer] = None
    linear_layer: Optional[LinearLayer] = None
    activation_layer: Optional[ActivationLayer] = None

class ResNetBlock(BaseModel):
    in_channels: int
    out_channels: int
    stride: int = 1
    activation_type: str 
    downsample: Optional[ConvLayer] = None
