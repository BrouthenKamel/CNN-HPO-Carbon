import numpy as np
import random
import math
from cooling import *


# rana nakhdmo b accuracy
class SimulatedAnnealing:
    def __init__(self, initial_temp, cooling_schedule, max_stagnation_iters = 5, stagnation_threshold=0.01):
        self.initial_temp = initial_temp
        self.cooling_schedule = cooling_schedule
        self.max_stagnation_iters = max_stagnation_iters
        self.stagnation_threshold = stagnation_threshold


    def optimize(self, model, hyperparameter_type, num_iterations):
        current_configuration = self.initialize_configuration(hyperparameter_type)
        current_score = self.evaluate(model, current_configuration)
        best_configuration = current_configuration
        best_score = current_score        
        archive = [(current_configuration, current_score)]
        current_temp = self.initial_temp
        stagnation_counter = 0
        

        for iteration in range(num_iterations):
            new_configuration = self.generate_neighborhood(current_configuration, hyperparameter_type)
            new_score = self.evaluate(model, new_configuration)
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
            
                    
                

            self.current_temp = self.cooling_schedule(self.current_temp, iteration)

        return best_configuration, best_score

    def initialize_configuration(self, hyperparameter_type):
        # hta ldrk the only idea is to just use the model's original hyperparameters
        # htan nzid nhawes 3la other methods (maybe use a heuristic ?)
        pass

    def generate_neighborhood(self, current_configuration, hyperparameter_type):
        # we use neighborhood generation ta ibrahim
        pass

    def evaluate(self, model, configuration):
        # we use surrogate model ta imed
        pass
    
    def cooling_schedule(self, current_temp, iteration):
        if self.cooling_schedule == 'exponential':
            return exponential_cooling(current_temp, iteration)
        elif self.cooling_schedule == 'linear':
            return linear_cooling(current_temp, 1, iteration)
        elif self.cooling_schedule == 'logarithmic':
            return logarithmic_cooling(current_temp, iteration)
        pass