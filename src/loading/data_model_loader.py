from matplotlib import pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision.models import alexnet, resnet50, vgg16


def load_dataset(dataset_name, data_dir='./data', download=True):
    """
    Loads the dataset.

    Args:
        dataset_name (str): The name of the dataset to load ('MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100').
        data_dir (str): The directory where the datasets will be stored or are located.
        download (bool): Whether to download the dataset if not found.

    Returns:
        torch.utils.data.Dataset: The loaded dataset.
        torchvision.transforms.Compose: The transformations applied to the dataset.

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
        dataset = torchvision.datasets.MNIST(root=data_dir, train=True, download=download, transform=transform)
    elif dataset_name == 'FashionMNIST':
        dataset = torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=download, transform=transform)
    elif dataset_name == 'CIFAR10':
        dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=download, transform=transform)
    elif dataset_name == 'CIFAR100':
        dataset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=download, transform=transform)

    print(f"Successfully loaded {dataset_name} dataset.")
    return dataset, transform


def load_model_architecture(model_name):
    """
    Loads the original, pre-defined architecture of the specified model

    Args:
        model_name (str): The name of the model architecture to load ('AlexNet', 'ResNet', 'VGG').

    Returns:
        torch.nn.Module: The loaded model architecture.

    Raises:
        ValueError: If the model_name is not supported.
    """
    if model_name == 'AlexNet':
        model = alexnet(weights=None)
    elif model_name == 'ResNet':
        model = resnet50(weights=None)
    elif model_name == 'VGG':
        model = vgg16(weights=None)
    else:
        raise ValueError(f"Unsupported model: {model_name}. Choose from 'AlexNet', 'ResNet', 'VGG'.")
    
    # save the model
    # torch.save(model, f'./models/{model_name}.pth')

    print(f"Successfully loaded {model_name} architecture.")
    return model


if __name__ == '__main__':
    try:
        mnist_dataset, mnist_transform = load_dataset('MNIST')
        print(f"MNIST dataset size: {len(mnist_dataset)}")

        if isinstance(mnist_dataset, torchvision.datasets.MNIST):
            print("Verification successful: Loaded object is a torchvision.datasets.MNIST.")
        else:
            print(f"Verification failed: Loaded object is of type {type(mnist_dataset)}, expected torchvision.datasets.MNIST.")

        if len(mnist_dataset) == 60000:
             print("Verification successful: Dataset size matches expected size for MNIST training set.")
        else:
             print(f"Verification failed: Dataset size is {len(mnist_dataset)}, expected 60000.")

        # for testing
        first_sample_image, first_sample_label = mnist_dataset[0]
        print(f"Shape of the first image sample: {first_sample_image.shape}")
        print(f"Label of the first image sample: {first_sample_label}")
        #show first sample image
        plt.imshow(first_sample_image.squeeze().numpy(), cmap='gray')
        plt.show()
        # ps: The image shape will be [C, H, W] after ToTensor() and normalization.
        # for MNIST it will be [1, 28, 28].

        # You can optionally create a DataLoader
        # mnist_dataloader = torch.utils.data.DataLoader(mnist_dataset, batch_size=64, shuffle=True)
    except ValueError as e:
        print(e)

    print("-" * 20) 

    # try:
    #     cifar10_dataset, cifar10_transform = load_dataset('CIFAR10', download=True) 
    #     print(f"CIFAR10 dataset size: {len(cifar10_dataset)}")

    #     if isinstance(cifar10_dataset, torchvision.datasets.CIFAR10):
    #         print("Verification successful: Loaded object is a torchvision.datasets.CIFAR10.")
    #     else:
    #         print(f"Verification failed: Loaded object is of type {type(cifar10_dataset)}, expected torchvision.datasets.CIFAR10.")

    #     if len(cifar10_dataset) == 50000:
    #          print("Verification successful: Dataset size matches expected size for CIFAR10 training set.")
    #     else:
    #          print(f"Verification failed: Dataset size is {len(cifar10_dataset)}, expected 50000.")

    #     # inspect a sample 
    #     # first_sample_image_cifar, first_sample_label_cifar = cifar10_dataset[0]
    #     # print(f"Shape of the first image sample (CIFAR10): {first_sample_image_cifar.shape}")
    #     # print(f"Label of the first image sample (CIFAR10): {first_sample_label_cifar}")
    #     # ps: the image shape will be [C, H, W] For CIFAR10, it will be [3, 32, 32].

    # except ValueError as e:
    #     print(e)

    # print("-" * 20) 

    try:
        alexnet_model = load_model_architecture('AlexNet')
        print("AlexNet model architecture loaded")
        print(alexnet_model)
    except ValueError as e:
        print(e)

    print("-" * 20) 

    try:
        resnet_model = load_model_architecture('ResNet')
        print("ResNet model architecture loaded")
        print(resnet_model)
    except ValueError as e:
        print(e)
