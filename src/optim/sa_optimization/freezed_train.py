import random
import time
import copy
from tqdm import tqdm

import torch
import torch.nn as nn
from src.loading.models.simple_CNN.space import SimpleCNNHPSpace
from src.loading.models.simple_CNN.hp import *
from src.loading.models.simple_CNN.config import SimpleCNNConfig
from src.loading.models.simple_CNN.model import SimpleCNN
from src.loading.data.loader import load_dataset
from src.schema.training import TrainingParams, OptimizerType
from src.training.train import train_model, count_parameters
from src.optim.sa_optimization.algorithm import SimulatedAnnealing  # adjust import path

# === Hyperparameter classes for the added CNN block ===
class ConvBlockHP:
    """
    Hyperparameter container for a single CNN block: conv layer params.
    """
    def __init__(self, filters: int, kernel_size: int, stride: int, padding: int):
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def to_dict(self):
        return {"filters": self.filters, "kernel_size": self.kernel_size, "stride": self.stride, "padding": self.padding}

class ConvBlockHPSpace:
    """
    Search space for a single CNN block's hyperparameters.
    """
    def __init__(self, filter_choices, kernel_sizes, strides, paddings):
        self.filter_choices = filter_choices
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.paddings = paddings

    def sample(self) -> ConvBlockHP:
        return ConvBlockHP(
            filters=random.choice(self.filter_choices),
            kernel_size=random.choice(self.kernel_sizes),
            stride=random.choice(self.strides),
            padding=random.choice(self.paddings)
        )

    def neighbor(self, hp: ConvBlockHP, ratio: float = 0.3, intensity: int = 1) -> ConvBlockHP:
        def perturb(choices, current):
            if random.random() > ratio:
                return current
            idx = choices.index(current)
            delta = random.choice([-intensity, intensity])
            new_idx = max(0, min(len(choices)-1, idx+delta))
            return choices[new_idx]
        return ConvBlockHP(
            filters=perturb(self.filter_choices, hp.filters),
            kernel_size=perturb(self.kernel_sizes, hp.kernel_size),
            stride=perturb(self.strides, hp.stride),
            padding=perturb(self.paddings, hp.padding)
        )

# 1. Prepare dataset and training params
data_training_params = TrainingParams(
    epochs=1,
    batch_size=64,
    learning_rate=1e-3,
    optimizer=OptimizerType.ADAM,
)
dataset = load_dataset('MNIST')

# 2. Define search space for full model
hp_space = SimpleCNNHPSpace(
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
        'epochs': [5,10],
        'batch_size': [16,32,64],
        'learning_rate': [1e-3,1e-4],
        'optimizer': ["ADAM","SGD"],
    }
)

# 3. Initial configuration for full model
initial_hp = SimpleCNNHP(
    initial_conv=ConvLayerHP(filters=32, kernel_size=3, stride=1, padding=1),
    cnn_block_hps=[
        CNNBlockHP(conv=ConvLayerHP(32,3,1,1), batch_norm=True, activation="RELU", pooling={"type":"MAX","kernel_size":2,"stride":2,"padding":0}),
        #CNNBlockHP(conv=ConvLayerHP(128,3,1,1), batch_norm=True, activation="RELU", pooling={"type":"MAX","kernel_size":2,"stride":2,"padding":0}),
    ],
    adaptive_pooling=4,
    mlp_block_hps=[
        MLPBlockHP(neurons=32, activation="RELU", dropout=0.5),
        #MLPBlockHP(neurons=128, activation="RELU", dropout=0.2),
    ],
    training={"epochs":1, "batch_size":64, "learning_rate":1e-3, "optimizer":"ADAM", "momentum":0.9}
)
print("Initial HP:", initial_hp.to_dict())

# 4. Define evaluation for full model
def evaluate(hp: SimpleCNNHP) -> float:
    config = SimpleCNNConfig.from_hp(hp)
    model = SimpleCNN(config, dataset.num_classes)
    print(f"Model Parameters: {count_parameters(model):.2f}M")
    start = time.time()
    results = train_model(model, dataset, data_training_params)
    elapsed = (time.time() - start) / 60
    acc = results.history.epochs[-1].test_accuracy
    print(f"Accuracy: {acc:.4f}, Time: {elapsed:.2f} min")
    return acc

# 5. Run SA on full model
# sa = SimulatedAnnealing(
#     init_configuration=initial_hp,
#     evaluator=evaluate,
#     initial_temp=100,
#     cooling_schedule="linear",
#     max_stagnation_iters=5,
#     stagnation_threshold=0.001,
#     search_space=hp_space,
#     neighborhood_generator_args={'ratio':0.3,'intensity':1}
# )
# print("Starting SA optimization...")
# (best_hp, best_score), history = sa.optimize(None, num_iterations=20)
# print("Best HP:", best_hp.to_dict())
# print(f"Best Score: {best_score:.4f}")

best_hp = initial_hp
# 6. Build final model, freeze and add new CNN block
final_config = SimpleCNNConfig.from_hp(best_hp)
model = SimpleCNN(final_config, dataset.num_classes)
for param in model.parameters(): param.requires_grad = False

# Define and append new block
new_conv = nn.Conv2d(
    in_channels=(model.features[-1].out_channels if isinstance(model.features[-1], nn.Conv2d)
                 else final_config.cnn_blocks[-1].conv_layer.filters),
    out_channels=64, kernel_size=3, stride=1, padding=1)
new_bn = nn.BatchNorm2d(64)
new_act = nn.ReLU(inplace=True)
new_pool = nn.MaxPool2d(2,2)
model.features = nn.Sequential(*list(model.features.children()), new_conv, new_bn, new_act, new_pool)
for layer in (new_conv, new_bn):
    for p in layer.parameters(): p.requires_grad = True

# 7. SA hyperparam search on new block only
# Use SimpleCNNHP to represent the full config, but only perturb the last CNN block
init_block_hp = copy.deepcopy(best_hp)
# Create and append new block hp matching new_conv
new_block_hp = CNNBlockHP(
    conv=ConvLayerHP(
        filters=(new_conv.out_channels if not isinstance(new_conv.out_channels, tuple) else new_conv.out_channels[0]),
        kernel_size=(new_conv.kernel_size if not isinstance(new_conv.kernel_size, tuple) else new_conv.kernel_size[0]),
        stride=(new_conv.stride if not isinstance(new_conv.stride, tuple) else new_conv.stride[0]),
        padding=(new_conv.padding if not isinstance(new_conv.padding, tuple) else new_conv.padding[0])
    ),
    batch_norm=True,
    activation="RELU",
    pooling={"type": "MAX", "kernel_size": 2, "stride": 2, "padding": 0}
)
init_block_hp.cnn_block_hps.append(new_block_hp)

final_config = SimpleCNNConfig.from_hp(init_block_hp)
model = SimpleCNN(final_config, dataset.num_classes)

# Define a custom BlockSpace that only perturbs the last block in a SimpleCNNHP
def block_neighbor(hp: SimpleCNNHP, ratio: float = 0.5, intensity: int = 1) -> SimpleCNNHP:
    new_hp = copy.deepcopy(hp)
    # perturb only the last CNNBlockHP
    block = new_hp.cnn_block_hps[-1]
    def perturb(choices, current):
        if random.random() > ratio:
            return current
        idx = choices.index(current)
        delta = random.choice([-intensity, intensity])
        new_idx = max(0, min(len(choices)-1, idx+delta))
        return choices[new_idx]
    # apply to conv parameters
    conv = block.conv
    conv.filters = perturb(hp_space.filter_choices, conv.filters)
    conv.kernel_size = perturb(hp_space.kernel_sizes, conv.kernel_size)
    conv.stride = perturb(hp_space.strides, conv.stride)
    conv.padding = perturb(hp_space.paddings, conv.padding)
    # apply to activation if applicable (optional)
    block.activation = random.choice(hp_space.activation_choices) if random.random() < ratio else block.activation
    # pooling unchanged for simplicity
    return new_hp


class BlockSpace:
    def neighbor(self, hp, ratio, intensity): return block_neighbor(hp, ratio, intensity)
block_space = BlockSpace()

sa_block = SimulatedAnnealing(
    init_configuration=init_block_hp,
    evaluator=evaluate,               # same full-evaluate, since only block changed
    initial_temp=10,
    cooling_schedule="linear",
    max_stagnation_iters=3,
    stagnation_threshold=0.001,
    search_space=block_space,
    neighborhood_generator_args={'ratio':0.5,'intensity':1}
)
(best_block_hp_full, best_block_score), _ = sa_block.optimize(None, num_iterations=1)
# Extract just the last block hp
best_block_hp = best_block_hp_full.cnn_block_hps[-1]
print("Best Block HP:", best_block_hp.to_dict(), "Score:", best_block_score)

# 8. Apply best and fine-tune
# update new_conv and new_bn
new_conv.out_channels = best_block_hp.filters
new_conv.kernel_size = best_block_hp.kernel_size
new_conv.stride = best_block_hp.stride
new_conv.padding = best_block_hp.padding
new_bn = nn.BatchNorm2d(best_block_hp.filters)
for p in model.parameters(): p.requires_grad = False
for p in new_conv.parameters(): p.requires_grad = True
for p in new_bn.parameters(): p.requires_grad = True
# fine-tune
ft_results = train_model(model, dataset, data_training_params)
print(f"Fine-tuned block accuracy: {ft_results.history.epochs[-1].test_accuracy:.4f}")
