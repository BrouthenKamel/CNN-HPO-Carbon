from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_mnist_data(batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize MNIST
    ])

    train_dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('.', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
