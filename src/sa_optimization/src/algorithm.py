import random
import torch.nn as nn
import math
from cooling import *
from src.neighborhood.src.neighborhood import ConfigurableCNN


# rana nakhdmo b accuracy
class SimulatedAnnealing:
    def __init__(self, initial_temp:int, cooling_schedule:str, search_space:dict, init_config:dict = None, max_stagnation_iters = 5, stagnation_threshold=0.01):
        self.initial_temp = initial_temp
        self.cooling_schedule = cooling_schedule
        self.max_stagnation_iters = max_stagnation_iters
        self.stagnation_threshold = stagnation_threshold
        self.search_space = search_space
        if init_config is None:
            self.init_config = self._initialize_configuration()
        else:
            self.init_config = init_config

    def optimize(self, hyperparameter_type, num_iterations):
        current_configuration = self.init_config
        current_score = self._evaluate(current_configuration)
        best_configuration = current_configuration
        best_score = current_score        
        archive = [(current_configuration, current_score)]
        current_temp = self.initial_temp
        stagnation_counter = 0
        

        for iteration in range(num_iterations):
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
            
                    
                

            self.current_temp = self._cooling_schedule(self.current_temp, iteration)

        return best_configuration, best_score

    def _initialize_configuration(self):
        # hta ldrk the only idea is to just use the model's original hyperparameters
        # htan nzid nhawes 3la other methods (maybe use a heuristic ?)
        pass

    def _generate_neighborhood(self, current_configuration, hyperparameter_type):
        neighborhood_generator = ConfigurableCNN(self.search_space, current_configuration)
        return neighborhood_generator.generate_neighbouring_config(to_modify=[hyperparameter_type])

    def _evaluate(self, configuration):
        # asna lkhdma ta imed 
        pass
    
    def _cooling_schedule(self, current_temp, iteration):
        if self.cooling_schedule == 'exponential':
            return exponential_cooling(current_temp, iteration)
        elif self.cooling_schedule == 'linear':
            return linear_cooling(current_temp, 1, iteration)
        elif self.cooling_schedule == 'logarithmic':
            return logarithmic_cooling(current_temp, iteration)
        pass