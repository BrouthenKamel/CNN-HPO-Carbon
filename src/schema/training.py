from enum import Enum
from typing import Optional, List
from pydantic import BaseModel

class OptimizerType(str, Enum):
    SGD = 'sgd'
    ADAM = 'adam'
    RMSPROP = 'rmsprop'
    ADAGRAD = 'adagrad'

class Training(BaseModel):
    epochs: int
    batch_size: int
    learning_rate: float
    optimizer: OptimizerType
    momentum: Optional[float] = None
    weight_decay: Optional[float] = None
