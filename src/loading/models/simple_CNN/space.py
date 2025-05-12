import torch
import torch.nn as nn
import random
from typing import List, Optional, Dict, Any

from src.loading.models.simple_CNN.hp import (
    SimpleCNNHP,
    ConvLayerHP,
    CNNBlockHP,
    MLPBlockHP
)
from src.loading.models.simple_CNN.config import SimpleCNNConfig
from src.schema.model import ModelArchitecture
from src.schema.layer import ActivationType, PoolingType

class SimpleCNNHPSpace:
    """
    Defines a hyperparameter search space for SimpleCNNHP, with sampling and neighbor-perturbation.
    """
    def __init__(
        self,
        filter_choices: List[int],
        kernel_sizes: List[int],
        strides: List[int],
        paddings: List[int],
        activation_choices: List[str],
        pooling_choices: List[Dict[str, Any]],
        adaptive_pool_sizes: List[int],
        mlp_neuron_choices: List[int],
        dropout_choices: List[float],
        training_space: Dict[str, List[Any]]
    ):
        self.filter_choices = filter_choices
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.paddings = paddings
        self.activation_choices = activation_choices
        self.pooling_choices = pooling_choices
        self.adaptive_pool_sizes = adaptive_pool_sizes
        self.mlp_neuron_choices = mlp_neuron_choices
        self.dropout_choices = dropout_choices
        self.training_space = training_space

    def sample(self) -> SimpleCNNHP:
        # Sample initial conv
        ic = None
        if random.random() < 0.5:
            ic = ConvLayerHP(
                filters=random.choice(self.filter_choices),
                kernel_size=random.choice(self.kernel_sizes),
                stride=random.choice(self.strides),
                padding=random.choice(self.paddings)
            )
        # Sample CNN blocks
        cnn_blocks = []
        for _ in range(random.randint(1, 4)):
            conv_hp = ConvLayerHP(
                filters=random.choice(self.filter_choices),
                kernel_size=random.choice(self.kernel_sizes),
                stride=random.choice(self.strides),
                padding=random.choice(self.paddings)
            )
            cnn_blocks.append(CNNBlockHP(
                conv=conv_hp,
                batch_norm=random.choice([True, False]),
                activation=random.choice(self.activation_choices),
                pooling=random.choice(self.pooling_choices)
            ))
        # Sample adaptive pooling size
        ap = random.choice(self.adaptive_pool_sizes)
        # Sample MLP blocks
        mlp_blocks = []
        for _ in range(random.randint(1, 3)):
            mlp_blocks.append(MLPBlockHP(
                neurons=random.choice(self.mlp_neuron_choices),
                activation=random.choice(self.activation_choices),
                dropout=random.choice(self.dropout_choices)
            ))
        # Sample training params
        tr = {
            'epochs': random.choice(self.training_space['epochs']),
            'batch_size': random.choice(self.training_space['batch_size']),
            'learning_rate': random.choice(self.training_space['learning_rate']),
            'optimizer': random.choice(self.training_space['optimizer']),
        }
        return SimpleCNNHP(
            initial_conv=ic,
            cnn_block_hps=cnn_blocks,
            adaptive_pooling=ap,
            mlp_block_hps=mlp_blocks,
            training=tr
        )

    def neighbor(
        self,
        hp: SimpleCNNHP,
        ratio: float = 0.3,
        intensity: int = 1
    ) -> SimpleCNNHP:
        def perturb(list_, current):
            if random.random() > ratio:
                return current
            idx = list_.index(current)
            delta = random.choice([-intensity, intensity])
            new_idx = max(0, min(len(list_)-1, idx+delta))
            return list_[new_idx]

        # Perturb initial conv
        ic = hp.initial_conv
        if ic:
            ic = ConvLayerHP(
                filters=perturb(self.filter_choices, ic.filters),
                kernel_size=perturb(self.kernel_sizes, ic.kernel_size),
                stride=perturb(self.strides, ic.stride),
                padding=perturb(self.paddings, ic.padding)
            )
        # Perturb blocks
        cnn_blocks = []
        for b in hp.cnn_block_hps:
            cnn_blocks.append(CNNBlockHP(
                conv=ConvLayerHP(
                    filters=perturb(self.filter_choices, b.conv.filters),
                    kernel_size=perturb(self.kernel_sizes, b.conv.kernel_size),
                    stride=perturb(self.strides, b.conv.stride),
                    padding=perturb(self.paddings, b.conv.padding)
                ),
                batch_norm=b.batch_norm,
                activation=perturb(self.activation_choices, b.activation),
                pooling=perturb(self.pooling_choices, b.pooling)
            ))
        # Perturb adaptive pooling
        ap = perturb(self.adaptive_pool_sizes, hp.adaptive_pooling)
        # Perturb MLP
        mlp_blocks = []
        for m in hp.mlp_block_hps:
            mlp_blocks.append(MLPBlockHP(
                neurons=perturb(self.mlp_neuron_choices, m.neurons),
                activation=perturb(self.activation_choices, m.activation),
                dropout=perturb(self.dropout_choices, m.dropout)
            ))
        # Training stays
        return SimpleCNNHP(
            initial_conv=ic,
            cnn_block_hps=cnn_blocks,
            adaptive_pooling=ap,
            mlp_block_hps=mlp_blocks,
            training=hp.training
        )


if __name__ == "__main__":
    # Define search space
    space = SimpleCNNHPSpace(
        filter_choices=[16, 32, 64, 128],
        kernel_sizes=[3,5],
        strides=[1,2],
        paddings=[0,1],
        activation_choices=["RELU","SIGMOID","TANH"],
        pooling_choices=[None, {"type": "MAX", "kernel_size": 2, "stride": 2, "padding": 0}],
        adaptive_pool_sizes=[1,2,4],
        mlp_neuron_choices=[64,128,256],
        dropout_choices=[0.1,0.2,0.5],
        training_space={
            'epochs':[5,10,20],
            'batch_size':[16,32,64],
            'learning_rate':[1e-3,1e-4],
            'optimizer':["ADAM","SGD"]
        }
    )
    hp = space.sample()
    print(hp.to_dict())
    neigh = space.neighbor(hp)
    print(neigh.to_dict())
