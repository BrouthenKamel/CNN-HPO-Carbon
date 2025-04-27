from src.loading.data.loader import load_dataset

for name in ['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100']:

    print(f"Loading dataset: {name}")
    
    dataset = load_dataset(name)

    print(f"Number of Classes: {dataset.num_classes}")
    print(f"Input Channels: {dataset.in_channels}")
    print(f"Train Dataset Size: {len(dataset.train_dataset)}")
    print(f"Test Dataset Size: {len(dataset.test_dataset)}")
    
    print("="*40)