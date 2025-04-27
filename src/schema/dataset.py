from torch.utils.data import Dataset as TorchDataset
from torchvision.transforms import Compose

from enum import Enum
# from pydantic import BaseModel

class DatasetName(str, Enum):
    MNIST = "MNIST"
    FashionMNIST = "FashionMNIST"
    CIFAR10 = "CIFAR10"
    CIFAR100 = "CIFAR100"

class Dataset():
    
    def __init__(self, name: str, train_dataset: TorchDataset, test_dataset: TorchDataset, transform: Compose, in_channels: int, num_classes: int):
        self.name = name
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.transform = transform
        self.in_channels = in_channels
        self.num_classes = num_classes
