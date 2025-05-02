import copy
import torch

from src.loading.models.mobilenet.hp import MobileNetHP, original_hp
from src.loading.models.mobilenet.space import MobileNetHPSpace
from src.training.train import train_model

from src.surrogate_modeling.rbf.model import GPRegressorSurrogate

def hill_climbing_optimization(
    initial_hp: MobileNetHP,
    hp_space: MobileNetHPSpace,
    surrogate_model: callable,
    actual_evaluation: callable,
    iterations: int = 10,
    neighbors_per_iteration: int = 10,
    actual_evaluations_per_iteration: int = 3,
    block_modification_ratio: float = 0.3,
    param_modification_ratio: float = 0.5,
    perturbation_intensity: int = 1,
    perturbation_strategy: str = "local"
):
    print("Starting Hill Climbing Optimization...")
    current_hp = initial_hp
    current_perf = actual_evaluation(current_hp)
    print(f"Initial performance: {current_perf:.4f}\n")

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

        print("Evaluating neighbors with surrogate model...")
        surrogate_preds = [
            (neighbor, surrogate_model(neighbor.get_flattened_representation()))
            for neighbor in neighbors
        ]
        surrogate_preds.sort(key=lambda x: x[1], reverse=True)

        top_candidates = surrogate_preds[:actual_evaluations_per_iteration * 3]
        print(f"Top {len(top_candidates)} candidates selected for actual evaluation")

        actual_evals = []
        for i, (hp, pred_perf) in enumerate(top_candidates):
            print(f"\nEvaluating candidate {i+1}/{len(top_candidates)} (Surrogate prediction: {pred_perf:.4f})")
            try:
                perf = actual_evaluation(hp)
                actual_evals.append((hp, perf))
                print(f"Actual performance: {perf:.4f}")
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    torch.cuda.empty_cache()
                    print("OOM Error: Skipping this model.")
                    continue
                else:
                    raise e
            if len(actual_evals) >= actual_evaluations_per_iteration:
                break

        if actual_evals:
            best_candidate, best_perf = max(actual_evals, key=lambda x: x[1])

            if best_perf > current_perf:
                print(f"New best model found with performance: {best_perf:.4f}")
                current_hp = best_candidate
                current_perf = best_perf
            else:
                print("No improvement found this iteration.")
        else:
            print("No valid candidates could be evaluated.")

        history.append((copy.deepcopy(current_hp), current_perf))
        print(f"End of iteration {iteration + 1}, best so far: {current_perf:.4f}\n")

    print("Optimization finished.")
    print(f"Best overall performance: {current_perf:.4f}")
    return current_hp, history