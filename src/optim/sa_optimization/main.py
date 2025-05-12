import random
import time
import copy
from tqdm import tqdm

import torch
from src.loading.models.simple_CNN.space import SimpleCNNHPSpace
from src.loading.models.simple_CNN.hp import *
from src.loading.models.simple_CNN.config import SimpleCNNConfig
from src.loading.models.simple_CNN.model import SimpleCNN
from src.loading.data.loader import load_dataset
from src.schema.training import TrainingParams, OptimizerType
from src.training.train import train_model, count_parameters
from src.surrogate_modeling.rbf.model import GPRegressorSurrogate
from src.optim.sa_optimization.algorithm import SimulatedAnnealing  # adjust import path to SimulatedAnnealing class

# 1. Prepare dataset and training params
training_params = TrainingParams(
    epochs=1,
    batch_size=64,
    learning_rate=1e-3,
    optimizer=OptimizerType.ADAM,
)
dataset = load_dataset('MNIST')  # or DatasetName.CIFAR10

# 2. Define search space
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

# 3. Initial configuration
initial_hp = SimpleCNNHP(
        initial_conv=ConvLayerHP(filters=32, kernel_size=3, stride=1, padding=1),
        cnn_block_hps=[
            CNNBlockHP(conv=ConvLayerHP(64, 3, 1, 1), batch_norm=True, activation="RELU", pooling={"type": "MAX", "kernel_size": 2, "stride": 2, "padding": 0}),
            #CNNBlockHP(conv=ConvLayerHP(128, 3, 1, 1), batch_norm=True, activation="RELU", pooling={"type": "MAX", "kernel_size": 2, "stride": 2, "padding": 0}),
        ],
        adaptive_pooling=4,
        mlp_block_hps=[
            MLPBlockHP(neurons=64, activation="RELU", dropout=0.5),
            #MLPBlockHP(neurons=128, activation="RELU", dropout=0.2),
        ],
        training={"epochs": 1, "batch_size": 64, "learning_rate": 1e-3, "optimizer": "ADAM", "momentum": 0.9}
    )
print("Initial HP:", initial_hp.to_dict())

# 4. Define evaluation function
def evaluate(hp: SimpleCNNHP) -> float:
    config = SimpleCNNConfig.from_hp(hp)
    model = SimpleCNN(config, dataset.num_classes)
    print(f"Model Parameters: {count_parameters(model):.2f}M")
    start = time.time()
    results = train_model(model, dataset, training_params)
    elapsed = (time.time() - start) / 60
    acc = results.history.epochs[-1].test_accuracy
    print(f"Accuracy: {acc:.4f}, Time: {elapsed:.2f} min")
    return acc

# 5. Instantiate and run Simulated Annealing
sa = SimulatedAnnealing(
    init_configuration=initial_hp,
    evaluator=evaluate,
    initial_temp=100,
    cooling_schedule="linear",
    max_stagnation_iters=5,
    stagnation_threshold=0.001,
    search_space=hp_space,
    neighborhood_generator_args={'ratio': 0.3, 'intensity': 1}
)

print("Starting SA optimization...")
(best_hp, best_score), history = sa.optimize(
    hyperparameter_type=None,
    num_iterations=3
)

print("Best HP:", best_hp.to_dict())
print(f"Best Score: {best_score:.4f}")
