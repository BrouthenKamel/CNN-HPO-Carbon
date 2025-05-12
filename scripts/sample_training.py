import time
import json
import os
import torch
import argparse

from src.loading.data.loader import load_dataset
from src.schema.dataset import DatasetName
from src.training.train import train_model, count_parameters
from src.schema.training import TrainingParams, OptimizerType

from src.loading.models.mobilenet.space import MobileNetHPSpace
from src.loading.models.mobilenet.config import MobileNetConfig
from src.loading.models.mobilenet.model import MobileNetV3Small
from src.loading.models.resnet18.space import ResNetHPSpace
from src.loading.models.resnet18.config import ResNetConfig
from src.loading.models.resnet18.model import ResNet

parser = argparse.ArgumentParser(description='Train a CNN model.')
parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'mobilenet'],
                    help='Choose the model architecture: resnet or mobilenet (default: resnet)')
args = parser.parse_args()

MODEL_CHOICE = args.model


base_dir = './dataset/'
os.makedirs(base_dir, exist_ok=True)

save_path = os.path.join(base_dir, f'training_record_{MODEL_CHOICE}.json')
model_save_dir = os.path.join(base_dir, f'models_{MODEL_CHOICE}')
os.makedirs(model_save_dir, exist_ok=True)


if os.path.exists(save_path):
    with open(save_path, 'r') as f:
        record = json.load(f)
else:
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

if MODEL_CHOICE == 'resnet':
    hp_space = ResNetHPSpace()
    ConfigClass = ResNetConfig
    ModelClass = ResNet
    model_name = "ResNet18"
elif MODEL_CHOICE == 'mobilenet':
    hp_space = MobileNetHPSpace()
    ConfigClass = MobileNetConfig
    ModelClass = MobileNetV3Small
    model_name = "MobileNetV3Small"
else:
    raise ValueError(f"Invalid model choice: {MODEL_CHOICE}. Choose 'resnet' or 'mobilenet'.")


def save_progress(record, path):
    with open(path, 'w') as f:
        json.dump(record, f, indent=4)

def save_model(model, path):
    torch.save(model.state_dict(), path)

print(f"Starting training loop for {model_name}...")

for i in range(n):
    print()

    while True:
        try:
            print(f"Sampling {model_name} Architecture...")
            hp = hp_space.sample()
            config = ConfigClass.from_hp(hp) 

            model = ModelClass(config, dataset.num_classes) 
            n_parameters = count_parameters(model)
            print(f"Model Instantiated successfully! Parameters: {n_parameters} Million")
            print(model)

            s = time.time()

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            res = train_model(model, dataset, training_params)

            print("Model summary:")
            # print(model) 
            e = time.time()
            break
        except Exception as e:
            if 'CUDA out of memory' in str(e):
                 print(f"GPU out of memory!")
            else:
                 print(f"An error occurred during training: {e}")
            print("Retrying...")
            time.sleep(2)


    training_time = round((e - s) / 60, 2)
    print(f"Training time: {training_time:.2f} minutes")

    record.append({
        'model_type': MODEL_CHOICE, 
        'hp': hp.to_dict(),
        'n_parameters': n_parameters,
        'history': res.history.model_dump(),
        'time': training_time
    })

    save_progress(record, save_path)

    save_model(model, os.path.join(model_save_dir, f'model_{i}.pth'))

    print("=" * 20)

print(f"Training completed for {model_name}!")
# print(record)
