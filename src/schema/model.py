from pydantic import BaseModel
from typing import Optional, List, Union

from schema.layer import AdaptivePoolingLayer, ConvLayer, PoolingLayer, BatchNormLayer
from schema.block import CNNBlock, MLPBlock, ResNetBlock
from schema.training import Training

class ModelArchitecture(BaseModel):
    cnn_blocks: List[CNNBlock] = []
    resnet_blocks: List[ResNetBlock] = []
    initial_conv_layer: Optional[ConvLayer] = None
    initial_bn_layer: Optional[BatchNormLayer] = None
    initial_activation_layer: Optional[dict] = None
    initial_pooling_layer: Optional[PoolingLayer] = None
    adaptive_pooling_layer: Optional[AdaptivePoolingLayer] = None
    mlp_blocks: List[MLPBlock] = []
    training: Training