from enum import Enum
from typing import Union, Optional
from pydantic import BaseModel

class ActivationType(str, Enum):
    RELU = 'relu'
    SIGMOID = 'sigmoid'
    TANH = 'tanh'
    SOFTMAX = 'softmax'
    
class ActivationLayer(BaseModel):
    type: ActivationType
    
class PaddingType(str, Enum):
    SAME = 'same'
    VALID = 'valid'

class ConvLayer(BaseModel):
    filters: int
    kernel_size: int
    stride: int
    padding: Union[int, PaddingType]
    
class PoolingType(str, Enum):
    MAX = 'max'
    AVG = 'avg'
    
class PoolingLayer(BaseModel):
    type: PoolingType
    kernel_size: int
    stride: int
    padding: Union[int, PaddingType]
    
class AdaptivePoolingLayer(BaseModel):
    type: PoolingType
    output_size: int
    
class DropoutLayer(BaseModel):
    rate: float
    
class LinearLayer(BaseModel):
    neurons: int

class BatchNormLayer(BaseModel):
    num_features: Optional[int] = None  # Make optional for initialization without knowing channels
    eps: float = 1e-5
    momentum: float = 0.1
    affine: bool = True
    track_running_stats: bool = True