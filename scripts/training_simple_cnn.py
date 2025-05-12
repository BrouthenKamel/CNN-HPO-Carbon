import time
import json
import os
import torch

from src.loading.data.loader import load_dataset
from src.schema.dataset import DatasetName
from src.training.train import train_model, count_parameters
from src.schema.training import TrainingParams, OptimizerType


from src.loading.models.simple_CNN.hp import SimpleCNNArchitecture
from src.neighborhood.neighboring import modify_value
from src.loading.models.simple_CNN.model import CNNModel

base_dir = './dataset/'
os.makedirs(base_dir, exist_ok=True)

save_path = os.path.join(base_dir, 'training_record_simple_cnn.json')

if os.path.exists(save_path):
    with open(save_path, 'r') as f:
        record = json.load(f)
else:
    record = []
    
record = []

training_params = TrainingParams(
    epochs=1,
    batch_size=64,
    learning_rate=0.001,
    optimizer=OptimizerType.ADAM,
    momentum=None,
    weight_decay=None,
)

n = 20

dataset_name = DatasetName.CIFAR10
dataset = load_dataset(dataset_name)
search_space = {
    "epochs": [10, 100, 200],
    "batch_size": [32, 64, 128, 256],
    "learning_rate": [0.0001, 0.001, 0.01, 0.0002, 0.0005, 0.0008, 0.002, 0.004, 0.005, 0.008],
    "momentum": [None],
    "weight_decay": [None],
    "filters": [32, 64, 96, 100, 128, 160, 192, 224, 256],
    "kernel_size": [2, 3, 4, 5, 6, 7],
    "stride": [1, 2, 3],
    "padding": [0, 1, 2, 3],
    "output_size": [3],
    "rate": [0.1, 0.2, 0.3, 0.4, 0.5],
    "neurons": [16, 32, 64, 128, 256, 512]
}

def save_progress(record, path):
    with open(path, 'w') as f:
        json.dump(record, f, indent=4)
        
def save_model(model, path):
    torch.save(model.state_dict(), path)

print("Sampling a Simple CNN Architecture...")
config = SimpleCNNArchitecture
# Randomizing the hp values
modify_value(config,block_modification_ratio=1,search_space=search_space,perturbation_intensity=10,perturbation_nature="Random")



# creating a nn.Module model from the config
model = CNNModel(config)
batch_size = config.training_params.batch_size
dummy_input = torch.randn(batch_size, 3, 32, 32)  # replace H and W with your input height and width
print(model)
model(dummy_input)
print(model)

n_parameters = count_parameters(model)
print(f"Model Instantiated successfully! Parameters: {n_parameters} Million")

s = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

res = train_model(model, dataset, training_params)

print("Model summary:")
print(model)
e = time.time()




# for i in range(n):
#     print()

#     while True:
#         try:
#             print("Sampling a Simple CNN Architecture...")
#             config = SimpleCNNArchitecture
#             # Randomizing the hp values
#             modify_value(config,block_modification_ratio=1,search_space=search_space,perturbation_intensity=10,perturbation_nature="Random")

#             # creating a nn.Module model from the config
#             model = CNNModel(config)
#             n_parameters = count_parameters(model)
#             print(f"Model Instantiated successfully! Parameters: {n_parameters} Million")

#             s = time.time()
            
#             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#             model.to(device)
            
#             res = train_model(model, dataset, training_params)
            
#             print("Model summary:")
#             print(model)
#             e = time.time()
#             break
#         except Exception as e:
#             print(e)
#             print(f"GPU out of memory!")
#             print("Retrying...")

#     training_time = round((e - s) / 60, 2)
#     print(f"Training time: {training_time:.2f} minutes")

#     record.append({
#         'hp': config.to_dict(),
#         'n_parameters': n_parameters,
#         'history': res.history.model_dump(),
#         'time': training_time
#     })

#     save_progress(record, save_path)
    
#     save_model(model, os.path.join(base_dir, f'model_{i}.pth'))

#     print(record)

#     print("=" * 20)

# print("Training completed!")
# print(record)
