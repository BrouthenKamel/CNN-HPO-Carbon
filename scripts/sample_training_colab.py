import time
import json
import os
import torch

from src.loading.data.loader import load_dataset
from src.schema.dataset import DatasetName
from src.training.train import train_model, count_parameters
from src.schema.training import TrainingParams, OptimizerType
from src.loading.models.mobilenet.space import MobileNetHPSpace
from src.loading.models.mobilenet.config import MobileNetConfig
from src.loading.models.mobilenet.model import MobileNetV3Small

base_dir = '/content/drive/MyDrive/mobilenet_2'
os.makedirs(base_dir, exist_ok=True)

save_path = os.path.join(base_dir, 'training_record.json')

if os.path.exists(save_path):
    with open(save_path, 'r') as f:
        record = json.load(f)
else:
    record = []
    
record = []

training_params = TrainingParams(
    epochs=5,
    batch_size=64,
    learning_rate=0.001,
    optimizer=OptimizerType.ADAM,
    momentum=None,
    weight_decay=None,
)

n = 5

dataset_name = DatasetName.CIFAR10
dataset = load_dataset(dataset_name)
hp_space = MobileNetHPSpace(num_blocks=11)

def save_progress(record, path):
    with open(path, 'w') as f:
        json.dump(record, f)
        
def save_model(model, path):
    torch.save(model.state_dict(), path)

for i in range(n):
    print()

    while True:
        try:
            print("Sampling MobileNet Architecture...")
            hp = hp_space.sample()
            config = MobileNetConfig.from_hp(hp)

            model = MobileNetV3Small(config, dataset.num_classes)
            n_parameters = count_parameters(model)
            print(f"Model Instantiated successfully! Parameters: {n_parameters} Million")

            s = time.time()
            res = train_model(model, dataset, training_params)
            
            print("Model Summary:")
            print(model)
            e = time.time()
            break
        except Exception as e:
            print(f"GPU out of memory!")
            print("Retrying...")

    training_time = round((e - s) / 60, 2)
    print(f"Training time: {training_time:.2f} minutes")

    record.append({
        'hp': hp.__dict__(),
        'n_parameters': n_parameters,
        'history': res.history.model_dump(),
        'time': training_time
    })

    save_progress(record, save_path)
    
    save_model(model, os.path.join(base_dir, f'model_{i}.pth'))

    print("=" * 20)

print("Training completed!")
print(record)
