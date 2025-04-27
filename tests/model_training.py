from src.loading.data.loader import load_dataset

from src.loading.models.model_builder import create_model
from src.loading.models.alexnet import AlexNetArchitecture
from src.loading.models.simple_cnn import SimpleCNNArchitecture 

from src.training.train import train_model

from src.schema.dataset import DatasetName

dataset = load_dataset(DatasetName.CIFAR10.value)
model = create_model(AlexNetArchitecture, in_channels=dataset.in_channels, num_classes=dataset.num_classes)

train_model(model, dataset, AlexNetArchitecture.training_params)