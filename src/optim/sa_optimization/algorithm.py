import random
import torch.nn as nn
import torch
import math
import copy
from tqdm import tqdm
import sys
import os

# Now imports should work
from .cooling import CoolingSchedule, exponential_cooling, linear_cooling, logarithmic_cooling
from src.neighborhood.neighboring import modify_value
from src.schema.model import ModelArchitecture
from src.schema.block import CNNBlock, MLPBlock
from src.schema.layer import ConvLayer, PoolingLayer, PoolingType, DropoutLayer, LinearLayer, ActivationLayer, ActivationType, PaddingType, AdaptivePoolingLayer
from src.schema.training import TrainingParams, OptimizerType
# from src.surrogate_modeling.models.alexnet.mnist.dt.inference import predict_from_config
from enum import Enum
# rana nakhdmo b accuracy


class SimulatedAnnealing:
    def __init__(
        self,
        init_configuration,
        evaluator:callable,
        initial_temp:int,
        cooling_schedule:CoolingSchedule,
        max_stagnation_iters = 5,
        stagnation_threshold=0.01,
        search_space = None,
        neighborhood_generator_args = None,
    ):
        
        self.initial_temp = initial_temp
        self.cooling_schedule = cooling_schedule
        self.max_stagnation_iters = max_stagnation_iters
        self.stagnation_threshold = stagnation_threshold
        self.init_configuration = init_configuration
        if (search_space):
            self.search_space = search_space
        # else:
            # self.search_space = modify_value
        self.evaluator = evaluator
        self.neighborhood_generator_args = neighborhood_generator_args
        
        
        


    def optimize(self, hyperparameter_type:list[str], num_iterations:int)-> tuple[ModelArchitecture, float]:
        print(f"starting simulated annealing optimization...")

        current_configuration = self.init_configuration
        current_score = self._evaluate(current_configuration)
        best_configuration = current_configuration
        best_score = current_score        
        archive = [(copy.deepcopy(current_configuration), current_score)]
        current_temp = self.initial_temp
        stagnation_counter = 0
        history = [(copy.deepcopy(current_configuration), current_score)]
        print(f"Initial score: {current_score}")
        print(f"Initial temperature: {current_temp}")
        

        for iteration in tqdm(range(num_iterations)):
            while True:
                new_configuration = self._generate_neighborhood(current_configuration, self.neighborhood_generator_args)
                print(f"evaluating neighbor at iter: {iteration}")
                try:
                    new_score = self._evaluate(new_configuration)
                    break  # successful evaluation → exit retry loop
                except Exception as e:
                    torch.cuda.empty_cache()
                    print(f"Error during evaluation: {e}. Retrying neighbor generation...")
                    continue  # stay in the same iteration, generate a new neighbor

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
                p = math.exp(-delta / current_temp)
                if random.random() < p:
                    archive.append((current_configuration, current_score))
                    current_configuration = new_configuration
                    current_score = new_score

            history.append((copy.deepcopy(current_configuration), current_score))
            self.current_temp = self._cooling_schedule(current_temp, iteration)


        return (best_configuration, best_score), history

    def _generate_neighborhood(self, current_configuration,  args = None, hyperparameter_type = None):
        return self.search_space.neighbor(
            current_configuration,
            # hyperparameter_type,
            **args
        )

    def _evaluate(self, configuration):
        return self.evaluator(configuration)
    
    def _cooling_schedule(self, current_temp, iteration):
        if self.cooling_schedule == CoolingSchedule.EXPONENTIAL:
            return exponential_cooling(current_temp, iteration)
        elif self.cooling_schedule == CoolingSchedule.LINEAR:
            return linear_cooling(current_temp, 1, iteration)
        elif self.cooling_schedule == CoolingSchedule.LOGARITHMIC:
            return logarithmic_cooling(current_temp, iteration)
        
        
        
        
