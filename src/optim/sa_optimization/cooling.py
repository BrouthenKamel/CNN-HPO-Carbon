from enum import Enum


def exponential_cooling(initial_temp, decay_rate):
    """Applies exponential cooling schedule."""
    return initial_temp * decay_rate

def linear_cooling(initial_temp, final_temp, num_iterations):
    """Applies linear cooling schedule."""
    return initial_temp - (initial_temp - final_temp) / num_iterations

def logarithmic_cooling(initial_temp, iteration):
    """Applies logarithmic cooling schedule."""
    return initial_temp / (1 + iteration)


class CoolingSchedule(Enum):
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    LOGARITHMIC = "logarithmic"