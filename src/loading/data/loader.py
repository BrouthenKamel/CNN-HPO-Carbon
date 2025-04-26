import os
import torchvision
import torchvision.transforms as transforms
import torch 

def load_dataset(dataset_name, data_dir='./data', download=True):
    """
    Loads the dataset. (Moved from data_model_loader.py)

    Args:
        dataset_name (str): The name of the dataset to load ('MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100').
        data_dir (str): The directory where the datasets will be stored or are located.
        download (bool): Whether to download the dataset if not found.

    Returns:
        torch.utils.data.Dataset: The loaded dataset (typically the training set).
        torchvision.transforms.Compose: The transformations applied to the dataset.
        int: Number of input channels.
        int: Number of classes.


    Raises:
        ValueError: If the dataset_name is not supported.
    """
    num_classes = 0
    in_channels = 0

    if dataset_name in ['MNIST', 'FashionMNIST']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)) 
        ])
        in_channels = 1
        num_classes = 10
    elif dataset_name in ['CIFAR10', 'CIFAR100']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
        ])
        in_channels = 3
        num_classes = 10 if dataset_name == 'CIFAR10' else 100
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Choose from 'MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100'.")

    dataset_class = None
    if dataset_name == 'MNIST':
        dataset_class = torchvision.datasets.MNIST
    elif dataset_name == 'FashionMNIST':
        dataset_class = torchvision.datasets.FashionMNIST
    elif dataset_name == 'CIFAR10':
        dataset_class = torchvision.datasets.CIFAR10
    elif dataset_name == 'CIFAR100':
        dataset_class = torchvision.datasets.CIFAR100

    dataset = dataset_class(root=data_dir, train=True, download=download, transform=transform)

    print(f"Successfully loaded {dataset_name} dataset (train={True}).")
    return dataset, transform, in_channels, num_classes
