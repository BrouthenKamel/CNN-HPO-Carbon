import os

from tqdm import tqdm

import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

def load_dataset(dataset_name: str, data_dir: str = './data', download: bool = True):
    """
    Loads the train and test datasets.

    Args:
        dataset_name (str): The name of the dataset to load ('MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100').
        data_dir (str): The directory where the datasets will be stored or are located.
        download (bool): Whether to download the dataset if not found.

    Returns:
        Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torchvision.transforms.Compose]:
            Train dataset, Test dataset, Transformations applied.

    Raises:
        ValueError: If the dataset_name is not supported.
    """
    
    if dataset_name in ['MNIST', 'FashionMNIST']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)) 
        ])
    elif dataset_name in ['CIFAR10', 'CIFAR100']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
        ])
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Choose from 'MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100'.")

    if dataset_name == 'MNIST':
        train_dataset = torchvision.datasets.MNIST(root=data_dir, train=True, download=download, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root=data_dir, train=False, download=download, transform=transform)
    elif dataset_name == 'FashionMNIST':
        train_dataset = torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=download, transform=transform)
        test_dataset = torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=download, transform=transform)
    elif dataset_name == 'CIFAR10':
        train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=download, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=download, transform=transform)
    elif dataset_name == 'CIFAR100':
        train_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=download, transform=transform)
        test_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=download, transform=transform)

    print(f"Successfully loaded {dataset_name} train and test datasets.")
    return train_dataset, test_dataset

if __name__ == "__main__":
    print("Loading datasets...\n")
    for dataset_name in ['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100']:
        print(f"Loading {dataset_name} dataset...")
        train_dataset, test_dataset = load_dataset(dataset_name)
        print(f"Number of training samples: {len(train_dataset)}")
        print(f"Number of testing samples: {len(test_dataset)}")
        print(f"Sample image shape: {train_dataset[0][0].shape}")
        print(f"Sample label: {train_dataset[0][1]}")
        print("-"*50)
    print("All datasets loaded successfully.")
