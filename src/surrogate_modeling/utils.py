import torch
import torch.nn as nn

def get_activation(name):
    return {
        'relu': nn.ReLU(inplace=True),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
        'leakyrelu': nn.LeakyReLU(inplace=True),
    }[name.lower()]

def get_pooling(pool_type, size, stride):
    pool_type = pool_type.lower()
    if pool_type == 'max':
        return nn.MaxPool2d(kernel_size=size, stride=stride)
    elif pool_type == 'avg':
        return nn.AvgPool2d(kernel_size=size, stride=stride)
    else:
        raise ValueError(f"Unsupported pooling type: {pool_type}")

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(output, 1)
        correct += (predicted == target).sum().item()
        total += target.size(0)

    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%")
    return avg_loss, accuracy

def evaluate(model, device, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            running_loss += loss.item()
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)

    avg_loss = running_loss / len(test_loader)
    accuracy = 100. * correct / total
    print(f"Test Loss = {avg_loss:.4f}, Test Accuracy = {accuracy:.2f}%")
    return avg_loss, accuracy