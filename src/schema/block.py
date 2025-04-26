from typing import Optional, List, Union

from pydantic import BaseModel

from src.schema.layer import ConvLayer, PoolingLayer, DropoutLayer, LinearLayer, ActivationLayer

class CNNBlock(BaseModel):
    conv_layer: ConvLayer
    activation_layer: Optional[ActivationLayer] = None
    pooling_layer: Optional[PoolingLayer] = None
    
class MLPBlock(BaseModel):
    dropout_layer: Optional[DropoutLayer] = None
    linear_layer: Optional[LinearLayer] = None
    activation_layer: Optional[ActivationLayer] = None
