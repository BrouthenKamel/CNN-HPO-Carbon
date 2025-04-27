import random
import torch.nn as nn
import math
from cooling import *
from src.neighborhood.src.neighboring import modify_value
from src.schema.model import ModelArchitecture
from tqdm import tqdm


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
            
                    
                

            self.current_temp = self._cooling_schedule(self.current_temp, iteration)

        return best_configuration, best_score

    def _generate_neighborhood(self, current_configuration, hyperparameter_type):
        for bloc in hyperparameter_type:
            element = getattr(current_configuration,bloc)
            modify_value(element)

    def _evaluate(self, configuration):
        # asna lkhdma ta imed 
        pass
    
    def _cooling_schedule(self, current_temp, iteration):
        if self.cooling_schedule == CoolingSchedule.EXPONENTIAL:
            return exponential_cooling(current_temp, iteration)
        elif self.cooling_schedule == CoolingSchedule.LINEAR:
            return linear_cooling(current_temp, 1, iteration)
        elif self.cooling_schedule == CoolingSchedule.LOGARITHMIC:
            return logarithmic_cooling(current_temp, iteration)