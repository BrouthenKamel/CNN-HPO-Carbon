from pydantic import BaseModel

from typing import Optional, List, Union

from src.schema.layer import AdaptivePoolingLayer
from src.schema.block import CNNBlock, MLPBlock
from src.schema.training import Training

class ModelArchitecture(BaseModel):
    cnn_blocks: List[CNNBlock]
    adaptive_pooling_layer: Optional[AdaptivePoolingLayer] = None
    mlp_blocks: List[MLPBlock]
    training: Training