import torch
import time

from src.loading.models.mobilenet.hp import MobileNetHP
from src.loading.models.mobilenet.space import MobileNetHPSpace
from src.training.train import train_model
from src.schema.training import TrainingParams, OptimizerType

from src.loading.models.mobilenet.config import MobileNetConfig
from src.loading.models.mobilenet.model import MobileNetV3Small

def evaluate(model, dataset, epochs: int, optimizer = None):
    training_params = TrainingParams(
        epochs=epochs,
        batch_size=64,
        learning_rate=0.005,
        optimizer=OptimizerType.ADAM,
        momentum=None,
        weight_decay=None,
    )
    
    start_time = time.time()
    train_results = train_model(model, dataset, training_params, optimizer=optimizer)
    eval_time = (time.time() - start_time) / 60

    test_accuracy = train_results.history.epochs[-1].test_accuracy

    print(f"Evaluated model with test_accuracy={test_accuracy:.4f}, time={eval_time:.2f} minutes")

    return train_results.model, train_results.optimizer, test_accuracy

def hill_climbing_optimization(
    initial_hp: MobileNetHP,
    hp_space: MobileNetHPSpace,
    dataset,
    iterations: int = 10,
    neighbors_per_iteration: int = 4,
    max_epochs: int = 20,
    block_modification_ratio: float = 0.3,
    param_modification_ratio: float = 0.5,
    perturbation_intensity: int = 1,
    perturbation_strategy: str = "local",
    freeze_blocks_until: int = 0,
):
    
    stage_schedule = [max_epochs // 4, max_epochs // 2, max_epochs]
    pretrained = freeze_blocks_until > 0

    # Initial evaluation
    current_hp = initial_hp
    config = MobileNetConfig.from_hp(current_hp)
    model = MobileNetV3Small(config, dataset.num_classes, pretrained=pretrained, freeze_blocks_until=freeze_blocks_until)
    optimizer = None

    _, optimizer, current_perf = evaluate(model, dataset, max_epochs, optimizer)
    history = []
    
    history.append({
        'iteration': 0,
        'best_hp': current_hp.to_dict(),
        'best_perf': current_perf,
    })

    for iter_idx in range(iterations):
        print(f"\n=== Iteration {iter_idx+1}/{iterations} ===")

        # Generate neighbors
        neighbors = [hp_space.neighbor(
            current_hp,
            block_modification_ratio,
            param_modification_ratio,
            perturbation_intensity,
            perturbation_strategy
        ) for _ in range(neighbors_per_iteration)]

        # Prepare candidate containers
        candidates = []
        for hp in neighbors:
            model = MobileNetV3Small(MobileNetConfig.from_hp(hp), dataset.num_classes, pretrained=pretrained, freeze_blocks_until=freeze_blocks_until)
            candidates.append({'hp': hp, 'model': model, 'optimizer': None, 'score': None})

        # Progressive staged training
        for stage_idx, stage_epochs in enumerate(stage_schedule[:-1]):
            print(f"\n→ Stage {stage_idx+1}: Training {len(candidates)} candidates @ {stage_epochs} epochs")

            next_candidates = []
            for i, candidate in enumerate(candidates):
                try:
                    model, optimizer, acc = evaluate(candidate['model'], dataset, stage_epochs, candidate['optimizer'])
                    candidate['model'] = model
                    candidate['optimizer'] = optimizer
                    candidate['score'] = acc
                    next_candidates.append(candidate)
                except Exception as e:
                    torch.cuda.empty_cache()
                    print(f"Candidate {i+1} failed:", e)

            if not next_candidates:
                print("All candidates failed.")
                break

            next_candidates.sort(key=lambda x: x['score'], reverse=True)
            candidates = next_candidates[:max(1, len(next_candidates) // 2)]

        # Final stage
        best_candidate = candidates[0]
        print(f"\n→ Final Stage: Retraining best candidate @ {max_epochs} epochs")

        try:
            final_model, final_opt, final_perf = evaluate(best_candidate['model'], dataset, max_epochs, best_candidate['optimizer'])
        except Exception as e:
            torch.cuda.empty_cache()
            print("Final evaluation failed:", e)
            final_perf = -1

        if final_perf > current_perf:
            print(f"New best model found! Accuracy: {final_perf:.4f}")
            current_hp = best_candidate['hp']
            current_perf = final_perf
        else:
            print("Final model did not outperform current best.")

        history.append({
            'iteration': iter_idx + 1,
            'best_hp': current_hp.to_dict(),
            'best_perf': current_perf,
        })
        print(f"Iteration {iter_idx+1} complete. Best so far: {current_perf:.4f}")

    print("\nOptimization finished.")
    return current_hp, history
