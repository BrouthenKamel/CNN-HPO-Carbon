import warnings
warnings.filterwarnings("ignore")

import time
import pandas as pd
from src.loading.models.mobilenet.config import MobileNetConfig
from src.loading.models.mobilenet.model import MobileNetV3Small
from src.loading.models.mobilenet.hp import MobileNetHP, original_hp
from src.loading.models.mobilenet.space import MobileNetHPSpace

from src.loading.models.simple_CNN.config import SimpleCNNConfig
from src.loading.models.simple_CNN.model import SimpleCNN
from src.loading.models.simple_CNN.hp import *
from src.loading.models.simple_CNN.space import SimpleCNNHPSpace

from src.training.train import train_model, count_parameters
from src.loading.data.loader import load_dataset
from src.schema.dataset import DatasetName
from src.schema.training import TrainingParams, OptimizerType
from src.surrogate_modeling.rbf.model import GPRegressorSurrogate
from .algorithm import SimulatedAnnealing

surrogate_model = GPRegressorSurrogate()
surrogate_model.load_model('src/surrogate_modeling/rbf/models/gpr.pkl')
print("Preparing training configuration and dataset...")
training_params = TrainingParams(
    epochs=10,
    batch_size=64,
    learning_rate=0.0025,
    optimizer=OptimizerType.ADAM,
    momentum=None,
    weight_decay=None,
)

dataset_name = DatasetName.CIFAR10
dataset = load_dataset(dataset_name)

print("Initializing hyperparameter space and surrogate model...")
# initial_hp = original_hp
# hp_space = MobileNetHPSpace()
initial_hp = SimpleCNNHP(
        initial_conv=ConvLayerHP(filters=32, kernel_size=3, stride=1, padding=1),
        cnn_block_hps=[
            CNNBlockHP(conv=ConvLayerHP(64, 3, 1, 1), batch_norm=True, activation="RELU", pooling={"type": "MAX", "kernel_size": 2, "stride": 2, "padding": 0}),
            CNNBlockHP(conv=ConvLayerHP(128, 3, 1, 1), batch_norm=True, activation="RELU", pooling={"type": "MAX", "kernel_size": 2, "stride": 2, "padding": 0}),
        ],
        adaptive_pooling=4,
        mlp_block_hps=[
            MLPBlockHP(neurons=256, activation="RELU", dropout=0.5),
            MLPBlockHP(neurons=128, activation="RELU", dropout=0.2),
        ],
        training={"epochs": 20, "batch_size": 64, "learning_rate": 1e-3, "optimizer": "ADAM", "momentum": 0.9}
    )

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
            'epochs':[5,10,20],
            'batch_size':[16,32,64],
            'learning_rate':[1e-3,1e-4],
            'optimizer':["ADAM","SGD"]
        })


# surrogate_model = GPRegressorSurrogate()
# surrogate_model.load_model('src/surrogate_modeling/rbf/models/gpr.pkl')

print("Defining actual evaluation function...")
# def actual_evaluation(hp: MobileNetHP):
#     config = MobileNetConfig.from_hp(hp)
#     model = MobileNetV3Small(config, dataset.num_classes)
#     print(f"Model Parameters: {count_parameters(model):.3f}M")
#     start_time = time.time()

#     train_results = train_model(model, dataset, training_params)
#     eval_time = (time.time() - start_time) / 60

#     test_accuracy = train_results.history.epochs[-1].test_accuracy

#     print(f"Evaluated HP with test_accuracy={test_accuracy:.4f}, time={eval_time:.2f} minutes")

#     return test_accuracy

def actual_evaluation(hp: SimpleCNNHP):
    config = SimpleCNNConfig.from_hp(hp)
    model = SimpleCNN(config, dataset.num_classes)
    print(f"Model Parameters: {count_parameters(model):.3f}M")
    start_time = time.time()

    train_results = train_model(model, dataset, training_params)
    eval_time = (time.time() - start_time) / 60

    test_accuracy = train_results.history.epochs[-1].test_accuracy

    print(f"Evaluated HP with test_accuracy={test_accuracy:.4f}, time={eval_time:.2f} minutes")

    return test_accuracy

# print("Starting hill climbing optimization...")
# best_hp, history = hill_climbing_optimization(
#     initial_hp=initial_hp,
#     hp_space=hp_space,
#     surrogate_model=lambda hp_flat: surrogate_model.predict(pd.DataFrame([hp_flat]))[0],
#     actual_evaluation=actual_evaluation,
#     iterations=10,
#     neighbors_per_iteration=10,
#     actual_evaluations_per_iteration=2,
#     block_modification_ratio=0.3,
#     param_modification_ratio=0.5,
#     perturbation_intensity=1,
#     perturbation_strategy="local"
# )

sa = SimulatedAnnealing(
    init_configuration=initial_hp,
    evaluator=actual_evaluation,
    initial_temp=100,
    cooling_schedule="linear",
    max_stagnation_iters=5,
    stagnation_threshold=0.01,
    search_space= hp_space,
    neighborhood_generator_args= {"block_modification_ratio":0.3,
     "param_modification_ratio":0.5,
     "perturbation_intensity":1,
     "perturbation_strategy":"local"
    }
)
print("=" * 40)
(best_hp, best_score), history = sa.optimize(
    hyperparameter_type=["block_modification_ratio", "param_modification_ratio", "perturbation_intensity", "perturbation_strategy"],
    num_iterations=20,
)
print("=" * 40)
print("Best Hyperparameters:")
print(best_hp.to_dict())
print("=" * 40)
print(f"Best Score: {best_score:.4f}")
print("=" * 40)
