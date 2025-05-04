import warnings
warnings.filterwarnings("ignore")

import json

from src.loading.models.mobilenet.hp import original_hp
from src.loading.models.mobilenet.space import MobileNetHPSpace
from src.loading.data.loader import load_dataset
from src.schema.dataset import DatasetName
from src.optim.hill_climbing.algorithm import hill_climbing_optimization

# Load dataset
dataset = load_dataset(DatasetName.CIFAR10)

# Initialize hyperparameter space
hp_space = MobileNetHPSpace(freeze_blocks_until=13)

# Run hill climbing optimization
best_hp, history = hill_climbing_optimization(
    initial_hp=original_hp,
    hp_space=hp_space,
    dataset=dataset,
    iterations=1,
    neighbors_per_iteration=4,
    max_epochs=4,
    block_modification_ratio=0.3,
    param_modification_ratio=0.5,
    perturbation_intensity=1,
    perturbation_strategy="local",
    freeze_blocks_until=13
)

# Print results
print("=" * 40)
print("Best Hyperparameters Found:")
print(best_hp.to_dict())
print("=" * 40)

with open('src/optim/hill_climbing/history.json', 'w') as f:
    json.dump(history, f, indent=4)

print("History saved to src/optim/hill_climbing/history.json")