import torchvision
import torchvision.transforms as transforms

from src.schema.dataset import Dataset

def load_dataset(dataset_name: str, data_dir: str = './data', augment: bool = False, augmentation_type: str = 'basic') -> tuple:

    if dataset_name in ['MNIST', 'FashionMNIST']:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        in_channels = 3
        num_classes = 10
        train_transform = transform
    elif dataset_name == 'CIFAR10':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])
        in_channels = 3
        num_classes = 10

        if augment:
            if augmentation_type == 'basic':
                train_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    # transforms.RandomCrop(224, padding=4),
                    # rotation and translation
                    transforms.RandomRotation(20),
                    # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2023, 0.1994, 0.2010))
                ])
            elif augmentation_type == 'auto':
                train_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2023, 0.1994, 0.2010))
                ])
            else:
                raise ValueError("augmentation_type must be either 'basic' or 'auto'")
        else:
            train_transform = transform
    elif dataset_name == 'CIFAR100':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409),
                                 (0.2673, 0.2564, 0.2762))
        ])
        in_channels = 3
        num_classes = 100
        train_transform = transform
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Choose from 'MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100'.")

    dataset_class = {
        'MNIST': torchvision.datasets.MNIST,
        'FashionMNIST': torchvision.datasets.FashionMNIST,
        'CIFAR10': torchvision.datasets.CIFAR10,
        'CIFAR100': torchvision.datasets.CIFAR100,
    }[dataset_name]

    train_dataset = dataset_class(root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset = dataset_class(root=data_dir, train=False, download=True, transform=transform)

    return Dataset(
        name=dataset_name,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        transform=transform,
        in_channels=in_channels,
        num_classes=num_classes
    )
