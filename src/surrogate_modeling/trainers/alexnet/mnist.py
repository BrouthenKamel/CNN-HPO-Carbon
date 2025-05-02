import pandas as pd
from architectures.alexnet import CustomAlexNet
from data_loaders.mnist import get_mnist_data
from torch import optim, nn
from utils import train, evaluate
import torch


def run_mnist(csv_path, row, result_csv_path, device='cuda'):
    # Load the configuration from CSV
    df = pd.read_csv(csv_path)
    config = df.iloc[row].to_dict() 

    # Model setup
    model = CustomAlexNet(csv_path=csv_path, input_channels=1, row=row)
    device_ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device_)
    model = nn.DataParallel(model, device_ids=[0, 1])
    
    # Get DataLoader for MNIST
    batch_size = int(config['batch_size'])
    train_loader, test_loader = get_mnist_data(batch_size)

    # Hyperparameters for training
    learning_rate = float(config['learning_rate'])
    num_epochs = int(config['num_epochs'])

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Optimizer selection
    # optimizer_type = config['optimizer'].lower()
    # if optimizer_type == 'sgd':
    #     optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    # elif optimizer_type == 'adam':
    #     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # else:
    #     raise ValueError(f"Unsupported optimizer: {optimizer_type}")

    # Loss function (cross-entropy for classification)
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    for epoch in range(1, num_epochs + 1):
        train_loss, train_accuracy = train(model, device_, train_loader, optimizer, criterion, epoch)
        
        
        # Optionally, save checkpoints every epoch
        torch.save(model.state_dict(), f"model_epoch_{epoch}.pt")

    test_loss, test_accuracy = evaluate(model, device_, test_loader, criterion)  

    result_df = pd.DataFrame({
        'row': [row],
        'learning_rate': [learning_rate],
        'num_epochs': [num_epochs],
        'batch_size': [batch_size],
        # 'optimizer': [optimizer_type],
        'train_accuracy': [train_accuracy],
        'test_accuracy': [test_accuracy]
    })
    
    # Append to the result CSV or create a new one
    result_df.to_csv(result_csv_path, mode='a', header=not pd.io.common.file_exists(result_csv_path), index=False)    

