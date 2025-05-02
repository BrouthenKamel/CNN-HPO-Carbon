import random
import torch.nn as nn
import math
import copy  # Added for deepcopy
from tqdm import tqdm
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# Now imports should work
from src.sa_optimization.src.cooling import CoolingSchedule, exponential_cooling, linear_cooling, logarithmic_cooling
from neighborhood.neighboring import modify_value
from src.schema.model import ModelArchitecture
from src.schema.block import CNNBlock, MLPBlock
from src.schema.layer import ConvLayer, PoolingLayer, PoolingType, DropoutLayer, LinearLayer, ActivationLayer, ActivationType, PaddingType, AdaptivePoolingLayer
from src.schema.training import Training, OptimizerType
from src.surrogate_modeling.models.alexnet.mnist.dt.inference import predict_from_config
# rana nakhdmo b accuracy
class SimulatedAnnealing:
    def __init__(self, init_configuration:ModelArchitecture, initial_temp:int, cooling_schedule:CoolingSchedule, max_stagnation_iters = 5, stagnation_threshold=0.01):
        self.initial_temp = initial_temp
        self.cooling_schedule = cooling_schedule
        self.max_stagnation_iters = max_stagnation_iters
        self.stagnation_threshold = stagnation_threshold
        self.init_configuration = init_configuration


    def optimize(self, hyperparameter_type:list[str], num_iterations:int)-> tuple[ModelArchitecture, float]:
        current_configuration = self.init_configuration
        current_score = self._evaluate(current_configuration)
        best_configuration = current_configuration
        best_score = current_score        
        archive = [(current_configuration, current_score)]
        current_temp = self.initial_temp
        stagnation_counter = 0
        

        for iteration in tqdm(range(num_iterations)):
            new_configuration = self._generate_neighborhood(current_configuration, hyperparameter_type)
            new_score = self._evaluate(new_configuration)
            delta = new_score - current_score
            if abs(delta) < self.stagnation_threshold:
                stagnation_counter += 1
                if stagnation_counter >= self.max_stagnation_iters:
                    weights = [w for _, w in archive]
                    chosen_config = random.choices(archive, weights=weights, k=1)[0]
                    archive.remove(chosen_config)
                    current_configuration = chosen_config[0]
                    current_score = chosen_config[1]
                    stagnation_counter = 0
                    
            
            if delta > 0:
                current_configuration = new_configuration
                current_score = new_score
                if current_score < best_score:
                    best_configuration = current_configuration
                    best_score = current_score
                    
            else:
                p = math.exp( - delta / current_temp)
                if random.random() < p:
                    archive.append((current_configuration, current_score))
                    current_configuration = new_configuration
                    current_score = new_score
            
                    
                

            self.current_temp = self._cooling_schedule(current_temp, iteration)

        return best_configuration, best_score

    def _generate_neighborhood(self, current_configuration, hyperparameter_type):
        current_config = copy.deepcopy(current_configuration)
        for bloc in hyperparameter_type:
            element = getattr(current_config,bloc)
            modify_value(element)
        return current_config

    def _evaluate(self, configuration):
        return predict_from_config(configuration)
    
    def _cooling_schedule(self, current_temp, iteration):
        if self.cooling_schedule == CoolingSchedule.EXPONENTIAL:
            return exponential_cooling(current_temp, iteration)
        elif self.cooling_schedule == CoolingSchedule.LINEAR:
            return linear_cooling(current_temp, 1, iteration)
        elif self.cooling_schedule == CoolingSchedule.LOGARITHMIC:
            return logarithmic_cooling(current_temp, iteration)
        
        
        
        
if __name__ == "__main__":
    # Example usage
    initial_temp = 1000
    cooling_schedule = CoolingSchedule.EXPONENTIAL
    max_stagnation_iters = 5
    stagnation_threshold = 0.01
    num_iterations = 100
    AlexNetArchitecture = ModelArchitecture(
    cnn_blocks=[
        CNNBlock(
            conv_layer=ConvLayer(filters=64, kernel_size=5, stride=1, padding=2),
            activation_layer=ActivationLayer(type=ActivationType.RELU.value),
            pooling_layer=PoolingLayer(type=PoolingType.MAX.value, kernel_size=2, stride=2, padding=0),
        ),
        CNNBlock(
            conv_layer=ConvLayer(filters=192, kernel_size=5, stride=1, padding=2),
            activation_layer=ActivationLayer(type=ActivationType.RELU.value),
            pooling_layer=PoolingLayer(type=PoolingType.MAX.value, kernel_size=2, stride=2, padding=0),
        ),
        CNNBlock(
            conv_layer=ConvLayer(filters=384, kernel_size=3, stride=1, padding=1),
            activation_layer=ActivationLayer(type=ActivationType.RELU.value),
        ),
        CNNBlock(
            conv_layer=ConvLayer(filters=256, kernel_size=3, stride=1, padding=1),
            activation_layer=ActivationLayer(type=ActivationType.RELU.value),
        ),
        CNNBlock(
            conv_layer=ConvLayer(filters=256, kernel_size=3, stride=1, padding=1),
            activation_layer=ActivationLayer(type=ActivationType.RELU.value),
            pooling_layer=PoolingLayer(type=PoolingType.MAX.value, kernel_size=2, stride=2, padding=0)
        ),
    ],
    adaptive_pooling_layer=AdaptivePoolingLayer(
        type=PoolingType.AVG.value,
        output_size=3
    ),
    mlp_blocks=[
        MLPBlock(
            dropout_layer=DropoutLayer(rate=0.5),
            linear_layer=LinearLayer(neurons=4096),
            activation_layer=ActivationLayer(type=ActivationType.RELU.value),
        ),
        MLPBlock(
            dropout_layer=DropoutLayer(rate=0.5),
            linear_layer=LinearLayer(neurons=(4096)),
            activation_layer=ActivationLayer(type=ActivationType.RELU.value),
        )
    ],
    training=Training(
        epochs=10,
        batch_size=32,
        learning_rate=0.001,
        optimizer=OptimizerType.SGD.value,
        momentum=None,
        weight_decay=None
    )
)
    
    sa = SimulatedAnnealing(AlexNetArchitecture, initial_temp, cooling_schedule, max_stagnation_iters, stagnation_threshold)
    best_configuration, best_score = sa.optimize(["cnn_blocks"], num_iterations)
    print("Best Configuration:", best_configuration)
    print("Best Score:", best_score)