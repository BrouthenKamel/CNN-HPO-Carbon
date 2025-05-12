import copy
import torch

from src.loading.models.mobilenet.hp import MobileNetHP
from src.loading.models.mobilenet.space import MobileNetHPSpace
from src.training.train import train_model
from src.loading.models.resnet18.hp import ResNetHP
from src.loading.models.resnet18.space import ResNetHPSpace
from src.loading.models.resnet18.model import ResNet
from src.loading.models.resnet18.config import ResNetConfig

def hill_climbing_optimization_staged_training(
    dataset,
    training_params,
    initial_hp: ResNetHP,
    hp_space: ResNetHPSpace,
    iterations: int = 10,
    neighbors_per_iteration: int = 10,
    block_modification_ratio: float = 0.3,
    param_modification_ratio: float = 0.5,
    perturbation_intensity: int = 1,
    perturbation_strategy: str = "local"
):
    """
    Hill climbing optimization with staged evaluation approach:
    1. Train all neighbors for 2 epochs
    2. Keep top 5 and train for 2 more epochs
    3. Keep top 3 and train for 10 more epochs
    4. Select the best model that improves upon current
    
    Args:
        initial_hp: Starting hyperparameters
        hp_space: Space to sample from
        dataset: Dataset to train on
        training_params: Base training parameters
        iterations: Number of hill climbing iterations
        neighbors_per_iteration: Number of neighbors to generate per iteration
        block_modification_ratio: Probability to modify block structure
        param_modification_ratio: Probability to modify parameters
        perturbation_intensity: Intensity of modifications
        perturbation_strategy: Strategy for perturbation ("local" or "global")
        
    Returns:
        best_hp: Best hyperparameters found
        history: List of (hp, performance) tuples throughout optimization
    """
    print("Starting Hill Climbing Optimization with Staged Training...")
    current_hp = initial_hp
    
    def train_and_evaluate(hp, epochs):
        try:
            config = ResNetConfig.from_hp(hp)
            model = ResNet(config, dataset.num_classes)
            
            custom_params = copy.deepcopy(training_params)
            custom_params.epochs = epochs
            
            metrics = train_model(model, dataset, custom_params)
            
            return model, metrics.test_accuracy[-1]
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                torch.cuda.empty_cache()
                print(f"OOM Error training model: {e}")
                return None, -1
            else:
                raise e

    print("Evaluating initial model...")
    initial_model, current_perf = train_and_evaluate(current_hp, epochs=1)  # 2+2+10 epochs
    if current_perf == -1:
        raise ValueError("Initial model training failed with OOM error.")
        
    print(f"Initial model performance: {current_perf:.4f}\n")
    history = [(copy.deepcopy(current_hp), current_perf)]

    for iteration in range(iterations):
        print(f"=== Iteration {iteration + 1}/{iterations} ===")
        
        print("Generating neighbors...")
        neighbors = [
            hp_space.neighbor(
                current_hp,
                block_modification_ratio,
                param_modification_ratio,
                perturbation_intensity,
                perturbation_strategy
            )
            for _ in range(neighbors_per_iteration)
        ]
        
        print("\nStage 1: Training all neighbors for 2 epochs...")
        stage1_results = []
        
        for i, neighbor_hp in enumerate(neighbors):
            print(f"Training neighbor {i+1}/{len(neighbors)} for 2 epochs")
            model, perf = train_and_evaluate(neighbor_hp, epochs=3)
            if perf != -1:  
                stage1_results.append((neighbor_hp, model, perf))
        
        stage1_results.sort(key=lambda x: x[2], reverse=True)
        top_5 = stage1_results[:5]
        print(f"\nStage 1 complete. Top 5 performances: {[f'{p:.4f}' for _, _, p in top_5]}")
        
        if len(top_5) == 0:
            print("No valid models after Stage 1. Skipping iteration.")
            continue
            
        print("\nStage 2: Training top 5 models for 2 more epochs...")
        stage2_results = []
        
        for i, (hp, _, _) in enumerate(top_5):
            print(f"Training candidate {i+1}/5 for 2 more epochs")
            model, perf = train_and_evaluate(hp, epochs=6)  # 4 = 2 + 2
            if perf != -1:
                stage2_results.append((hp, model, perf))
        
        stage2_results.sort(key=lambda x: x[2], reverse=True)
        top_3 = stage2_results[:3]
        print(f"\nStage 2 complete. Top 3 performances: {[f'{p:.4f}' for _, _, p in top_3]}")
        
        if len(top_3) == 0:
            print("No valid models after Stage 2. Skipping iteration.")
            continue
        
        print("\nStage 3: Training top 3 models for 10 more epochs...")
        stage3_results = []
        
        for i, (hp, _, _) in enumerate(top_3):
            print(f"Training candidate {i+1}/3 for full 14 epochs")
            model, perf = train_and_evaluate(hp, epochs=10)  # 14 = 2 + 2 + 10
            if perf != -1:
                stage3_results.append((hp, perf))
        
        if stage3_results:
            best_hp, best_perf = max(stage3_results, key=lambda x: x[1])
            
            print(f"\nStage 3 complete. Best performance: {best_perf:.4f}")
            
            if best_perf > current_perf:
                print(f"New best model found with improvement: +{best_perf - current_perf:.4f}")
                current_hp = best_hp
                current_perf = best_perf
            else:
                print(f"No improvement found. Current best: {current_perf:.4f}")
        else:
            print("No valid models after Stage 3. Keeping current best.")
            
        history.append((copy.deepcopy(current_hp), current_perf))
        print(f"End of iteration {iteration + 1}, best so far: {current_perf:.4f}\n")

    print("Optimization finished.")
    print(f"Best overall performance: {current_perf:.4f}")
    return current_hp, history
