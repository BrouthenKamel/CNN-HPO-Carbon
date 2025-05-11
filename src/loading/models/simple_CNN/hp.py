from typing import List, Optional, Union
import random

# Hyperparameter container classes for Simple CNN
class ConvLayerHP:
    def __init__(self, filters: int, kernel_size: int, stride: int, padding: int):
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def to_dict(self):
        return {"filters": self.filters, "kernel_size": self.kernel_size, "stride": self.stride, "padding": self.padding}

class CNNBlockHP:
    def __init__(self, conv: ConvLayerHP, batch_norm: bool, activation: Optional[str], pooling: Optional[dict]):
        self.conv = conv
        self.batch_norm = batch_norm
        self.activation = activation
        self.pooling = pooling  # {"type": str, "kernel_size": int, "stride": int, "padding": int}

    def to_dict(self):
        return {"conv": self.conv.to_dict(), "batch_norm": self.batch_norm, "activation": self.activation, "pooling": self.pooling}

class MLPBlockHP:
    def __init__(self, neurons: int, activation: Optional[str], dropout: Optional[float]):
        self.neurons = neurons
        self.activation = activation
        self.dropout = dropout

    def to_dict(self):
        return {"neurons": self.neurons, "activation": self.activation, "dropout": self.dropout}

class SimpleCNNHP:
    def __init__(
        self,
        initial_conv: Optional[ConvLayerHP],
        cnn_block_hps: List[CNNBlockHP],
        adaptive_pooling: Optional[int],
        mlp_block_hps: List[MLPBlockHP],
        training: dict
    ):
        self.initial_conv = initial_conv
        self.cnn_block_hps = cnn_block_hps
        self.adaptive_pooling = adaptive_pooling
        self.mlp_block_hps = mlp_block_hps
        self.training = training  # {"epochs", "batch_size", "learning_rate", "optimizer", ...}

    def to_dict(self):
        return {
            "initial_conv": self.initial_conv.to_dict() if self.initial_conv else None,
            "cnn_blocks": [b.to_dict() for b in self.cnn_block_hps],
            "adaptive_pooling": self.adaptive_pooling,
            "mlp_blocks": [m.to_dict() for m in self.mlp_block_hps],
            "training": self.training
        }