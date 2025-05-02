import pandas as pd
import random

def generate_random_config(num_configs, output_csv='alexnet_random_configs.csv'):
    data = []

    for _ in range(num_configs):
        config = {}

        # === CLB Hyperparameters (5 blocks) ===
        for i in range(1, 6):
            config[f'clb{i}_kernels'] = random.choice([32, 64, 96, 100, 128, 160, 192, 224, 256])
            config[f'clb{i}_kernel_size'] = random.choice([3, 5, 7])
            config[f'clb{i}_stride'] = random.choice([1, 4])
            config[f'clb{i}_activation'] = random.choice(["relu", "tanh", "sigmoid", "leakyrelu"])
            if i in [1, 2, 5]:  # Only these have pooling
                config[f'clb{i}_pool_type'] = random.choice(['max', 'avg'])
                config[f'clb{i}_pool_size'] = random.choice([2, 3])
                config[f'clb{i}_pool_stride'] = random.choice([1, 2])
            # if i in [1, 2]:  # Only 1 & 2 have LRN
            #     config[f'clb{i}_lrn_k'] = 2
            #     config[f'clb{i}_lrn_n'] = 5
            #     config[f'clb{i}_lrn_alpha'] = 1e-4
            #     config[f'clb{i}_lrn_beta'] = 0.75

        # === FC Hyperparameters (5 layers) ===
        fc_sizes = []
        for i in range(1, 4):  # First 3 FC layers (before final classification)
            if i == 1 or fc_sizes[-1] != 0:
                neurons = random.choice([0, 16, 32, 64, 128, 256, 512])
            else:
                neurons = 0
            fc_sizes.append(neurons)
            config[f'fc{i}_neurons'] = neurons
            if neurons != 0:
                # config[f'fc{i}_init'] = random.choice(['xavier', 'normal'])
                config[f'fc{i}_dropout'] = random.choice([0.1, 0.2, 0.3, 0.4, 0.5])
                # config[f'fc{i}_regularization'] = random.choice(['l2', 'none'])

        # Final classification layers
        config['fc4_neurons'] = random.choice([16, 32, 64, 128, 256, 512])
        # config['fc4_init'] = random.choice(['xavier', 'normal'])
        config['fc4_dropout'] = random.choice([0.1, 0.2, 0.3, 0.4, 0.5])
        # config['fc4_regularization'] = random.choice(['l2', 'none'])


        # === Training Hyperparameters ===
        config['learning_rate'] = random.choice([0.0001, 0.0002, 0.0005, 0.0008, 0.001, 0.002, 0.004, 0.005, 0.008, 0.01])
        config['num_epochs'] = random.choice([3])
        config['batch_size'] = random.choice([32, 64])
        # config['optimizer'] = random.choice(['adam'])

        data.append(config)

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Generated {num_configs} configurations and saved to {output_csv}")

# Example usage
generate_random_config(300, output_csv="./datasets/generated/alexnet_random_configs_v7.csv")
