import torchvision
import torchvision.transforms as transforms

from src.schema.dataset import Dataset

def load_dataset(dataset_name: str, data_dir: str = './data') -> tuple:

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

    train_dataset = dataset_class(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = dataset_class(root=data_dir, train=False, download=True, transform=transform)    
    
    return Dataset(
        name=dataset_name,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        transform=transform,
        in_channels=in_channels,
        num_classes=num_classes
    )
