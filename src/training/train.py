import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import time
import os

from tqdm import tqdm

from src.loading.data.loader import load_dataset
from src.loading.models.model_builder import create_model
from src.schema.training import OptimizerType, TrainingParams, Epoch, History, TrainingResult
from src.schema.dataset import Dataset

# from src.loading.models.alexnet import AlexNetArchitecture
# from src.loading.models.model_builder import load_model_architecture

def count_parameters(model: torch.nn.Module) -> int:
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n / 1e6

def get_optimizer(model: nn.Module, training_params: TrainingParams):
    
    kwargs = {}
    if training_params.learning_rate is not None:
        kwargs['lr'] = training_params.learning_rate
    if training_params.momentum is not None:
        kwargs['momentum'] = training_params.momentum
    if training_params.weight_decay is not None:
        kwargs['weight_decay'] = training_params.weight_decay
        
    if training_params.optimizer == OptimizerType.SGD:
        return optim.SGD(model.parameters(), **kwargs)
    elif training_params.optimizer == OptimizerType.ADAM:
        return optim.Adam(model.parameters(), **kwargs)
    elif training_params.optimizer == OptimizerType.RMSPROP:
        return optim.RMSprop(model.parameters(), **kwargs)
    elif training_params.optimizer == OptimizerType.ADGARAD:
        return optim.Adagrad(model.parameters(), **kwargs)

def evaluate_model(model: nn.Module, data_loader, criterion, device):
    """
    Evaluates the model on the provided data loader with a tqdm progress bar.

    Args:
        model (nn.Module): The model to evaluate.
        data_loader (DataLoader): DataLoader for the dataset to evaluate on.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to perform computation on.

    Returns:
        Tuple[float, float]: Average loss and accuracy in percentage.
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="Evaluating", leave=False)
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            avg_loss = total_loss / total if total > 0 else 0
            accuracy = 100 * correct / total if total > 0 else 0

            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{accuracy:.2f}%'
            })

    avg_loss = total_loss / total
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def train_model(model: nn.Module, dataset, training_params: TrainingParams, patience: int = 5, optimizer: torch.optim.Optimizer = None) -> TrainingResult:
    """
    Trains the model on the provided dataset using the specified training parameters.

    Args:
        model (nn.Module): The model to train.
        dataset: An object with 'train_dataset' and 'test_dataset' attributes.
        training_params: An object with 'batch_size', 'epochs', and other training parameters.
    """
    print(f"Model parameters: {count_parameters(model):.3f} Million")
    torch.autograd.set_detect_anomaly(True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = get_optimizer(model, training_params)

    train_loader = DataLoader(dataset.train_dataset, batch_size=training_params.batch_size, shuffle=True)
    test_loader = DataLoader(dataset.test_dataset, batch_size=training_params.batch_size, shuffle=False)

    history = History(epochs=[])

    best_accuracy = 0.0
    best_model_state = None
    best_optimizer_state = None
    epochs_since_improvement = 0

    for epoch in range(training_params.epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{training_params.epochs}", leave=False)
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            progress_bar.set_postfix({
                'loss': f'{train_loss / train_total:.4f}',
                'acc': f'{100 * train_correct / train_total:.2f}%'
            })

        avg_train_loss = train_loss / train_total
        train_accuracy = 100 * train_correct / train_total
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Train Accuracy: {train_accuracy:.2f}%")

        test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
        print(f"Epoch {epoch+1} | Test Loss:  {test_loss:.4f} | Test Accuracy:  {test_accuracy:.2f}%")

        history.record_epoch(
            epoch=epoch+1,
            train_loss=avg_train_loss,
            test_loss=test_loss,
            train_accuracy=train_accuracy,
            test_accuracy=test_accuracy
        )

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model_state = model.state_dict()
            best_optimizer_state = optimizer.state_dict()
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= patience:
                print(f"No improvement in {patience} epochs. Early stopping.")
                break

        print('-' * 20)

    torch.cuda.empty_cache()
    print(f"Finished Training. Best Test Accuracy: {best_accuracy:.2f}%")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        optimizer.load_state_dict(best_optimizer_state)

    return TrainingResult(model, history, best_accuracy, optimizer)

# def train_model_with_args(args):
#     """
#     Trains a specified model on a specified dataset, using command-line args
#     to potentially override schema defaults.

#     Args:
#         args (argparse.Namespace): Parsed command-line arguments.
#     """
#     model_name = args.model
#     dataset_name = args.dataset
#     data_dir = args.data_dir
#     val_split = args.val_split
#     test_split = args.test_split
#     save_path_dir = args.save_path

#     print(f"Starting training process for Model: {model_name} on Dataset: {dataset_name}")

#     print("Loading dataset...")
#     try:
#         full_dataset, transform, in_channels, num_classes = load_dataset(dataset_name, data_dir=data_dir)
#         print(f"Dataset loaded: {len(full_dataset)} samples, In Channels: {in_channels}, Num Classes: {num_classes}")
#     except ValueError as e:
#         print(f"Error loading dataset: {e}")
#         return
#     except Exception as e:
#         print(f"An unexpected error occurred during dataset loading: {e}")
#         return

#     total_size = len(full_dataset)
#     test_size = int(test_split * total_size)
#     val_size = int(val_split * (total_size - test_size))
#     train_size = total_size - test_size - val_size

#     if train_size <= 0 or val_size <= 0 or test_size <= 0:
#         print(f"Error: Dataset split resulted in zero samples for one or more sets.")
#         print(f"Total: {total_size}, Train: {train_size}, Val: {val_size}, Test: {test_size}")
#         print("Adjust splits or check dataset size.")
#         return

#     try:
#         train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
#         print(f"Dataset split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
#     except Exception as e:
#         print(f"Error splitting dataset: {e}")
#         return

#     print("Loading model architecture schema...")
#     try:
#         architecture_schema = load_model_architecture(model_name)
#         schema_training_params: TrainingParams = architecture_schema.training
#         print(f"Schema defaults: Epochs={schema_training_params.epochs}, Batch={schema_training_params.batch_size}, LR={schema_training_params.learning_rate}, Optim={schema_training_params.optimizer.value}, Momentum={schema_training_params.momentum}, WeightDecay={schema_training_params.weight_decay}")
#     except ValueError as e:
#         print(f"Error loading model schema: {e}")
#         return
#     except Exception as e:
#         print(f"An unexpected error occurred loading model schema: {e}")
#         return

#     epochs = args.epochs if args.epochs is not None else schema_training_params.epochs
#     batch_size = args.batch_size if args.batch_size is not None else schema_training_params.batch_size
#     learning_rate = args.lr if args.lr is not None else schema_training_params.learning_rate
#     optimizer_str = args.optimizer if args.optimizer is not None else schema_training_params.optimizer.value
#     momentum = args.momentum if args.momentum is not None else schema_training_params.momentum
#     weight_decay = args.weight_decay if args.weight_decay is not None else schema_training_params.weight_decay

#     print(f"Effective Params: Epochs={epochs}, Batch={batch_size}, LR={learning_rate}, Optim={optimizer_str}, Momentum={momentum}, WeightDecay={weight_decay}")

#     print("Building model...")
#     try:
#         model = create_model(architecture_schema, in_channels=in_channels, num_classes=num_classes)
#     except Exception as e:
#         print(f"Error building model from schema: {e}")
#         return

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     print(f"Model built and moved to device: {device}")

#     print("Attempting to initialize lazy layers (if any)...")
#     try:
#         sample_img, _ = full_dataset[0]
#         dummy_batch_size = 2 
#         dummy_input = sample_img.unsqueeze(0).repeat(dummy_batch_size, 1, 1, 1).to(device)

#         model.eval() 
#         with torch.no_grad():
#             _ = model(dummy_input)
#         model.train() 
#         print("Lazy layers initialized successfully.")
#     except Exception as e:
#         print(f"Warning: Error during lazy layer initialization: {e}")
#         print("Proceeding, but this might cause issues if model has lazy layers.")

#     criterion = nn.CrossEntropyLoss()
#     try:
#         optimizer = get_optimizer(
#             optimizer_str,
#             model.parameters(),
#             lr=learning_rate,
#             momentum=momentum,
#             weight_decay=weight_decay
#         )
#     except ValueError as e:
#         print(f"Error creating optimizer: {e}")
#         return

#     num_workers = 2 if device.type == 'cuda' else 0 
#     try:
#         train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False)
#         val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False)
#         test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False)
#         print("Optimizer, Loss, and DataLoaders configured.")
#     except Exception as e:
#         print(f"Error creating DataLoaders: {e}")
#         return

#     print(f"Starting training loop for {epochs} epochs...")
#     start_time = time.time()
#     best_val_accuracy = 0.0
#     best_model_path = ""

#     for epoch in range(epochs):
#         epoch_start_time = time.time()
#         model.train() 
#         running_loss = 0.0
#         correct_train = 0
#         total_train = 0

#         for i, data in enumerate(train_loader):
#             try:
#                 inputs, labels = data[0].to(device), data[1].to(device)

#                 optimizer.zero_grad()

#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)
#                 loss.backward()
#                 optimizer.step()

#                 running_loss += loss.item()
#                 _, predicted = torch.max(outputs.data, 1)
#                 total_train += labels.size(0)
#                 correct_train += (predicted == labels).sum().item()

#                 if (i + 1) % 100 == 0:
#                     print(f'  Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

#             except Exception as batch_error:
#                 print(f"Error during training batch {i+1} in epoch {epoch+1}: {batch_error}")
#                 continue 

#         epoch_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0
#         epoch_acc_train = (100 * correct_train / total_train) if total_train > 0 else 0

#         model.eval() 
#         correct_val = 0
#         total_val = 0
#         val_loss = 0.0
#         with torch.no_grad():
#             for data in val_loader:
#                 try:
#                     images, labels = data[0].to(device), data[1].to(device)
#                     outputs = model(images)
#                     loss = criterion(outputs, labels)
#                     val_loss += loss.item()
#                     _, predicted = torch.max(outputs.data, 1)
#                     total_val += labels.size(0)
#                     correct_val += (predicted == labels).sum().item()
#                 except Exception as val_batch_error:
#                     print(f"Error during validation batch: {val_batch_error}")
#                     continue 

#         epoch_acc_val = (100 * correct_val / total_val) if total_val > 0 else 0
#         epoch_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
#         epoch_end_time = time.time()

#         print(f"Epoch {epoch+1}/{epochs} Summary: ")
#         print(f"  Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc_train:.2f}%")
#         print(f"  Val Loss:   {epoch_val_loss:.4f}, Val Acc:   {epoch_acc_val:.2f}%")
#         print(f"  Epoch Time: {epoch_end_time - epoch_start_time:.2f} seconds")

#         if epoch_acc_val > best_val_accuracy:
#             best_val_accuracy = epoch_acc_val
#             os.makedirs(save_path_dir, exist_ok=True) 
#             current_best_path = os.path.join(save_path_dir, f'{model_name}_{dataset_name}_best.pth')
#             try:
#                 torch.save(model.state_dict(), current_best_path)
#                 best_model_path = current_best_path 
#                 print(f"  Best validation accuracy improved to {best_val_accuracy:.2f}%. Model saved to {best_model_path}")
#             except Exception as save_error:
#                 print(f"Error saving model: {save_error}")

#     total_training_time = time.time() - start_time
#     print(f"Finished Training. Total time: {total_training_time // 60:.0f}m {total_training_time % 60:.0f}s")

#     print("\nLoading best model for final test evaluation...")
#     if os.path.exists(best_model_path):
#         print(f"Loading model from: {best_model_path}")
#         try:
#             final_model = create_model(architecture_schema, in_channels=in_channels, num_classes=num_classes)
#             final_model.load_state_dict(torch.load(best_model_path, map_location=device)) 
#             final_model.to(device)
#             final_model.eval()

#             correct_test = 0
#             total_test = 0
#             with torch.no_grad():
#                 for data in test_loader:
#                     try:
#                         images, labels = data[0].to(device), data[1].to(device)
#                         outputs = final_model(images)
#                         _, predicted = torch.max(outputs.data, 1)
#                         total_test += labels.size(0)
#                         correct_test += (predicted == labels).sum().item()
#                     except Exception as test_batch_error:
#                          print(f"Error during testing batch: {test_batch_error}")
#                          continue 

#             test_accuracy = (100 * correct_test / total_test) if total_test > 0 else 0
#             print(f"Final Test Accuracy on {len(test_dataset)} samples: {test_accuracy:.2f}%")
#         except Exception as load_error:
#             print(f"Error loading or evaluating best model: {load_error}")
#     else:
#         print(f"Could not find best model at expected path: {best_model_path}. Skipping final test evaluation.")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Train a CNN model defined by schema on a specified dataset.")

#     parser.add_argument("--model", type=str, required=True, # ['AlexNet', 'VGG', 'ResNet'],
#                         help="Name of the model architecture to train (e.g., AlexNet, VGG, ResNet).")
#     parser.add_argument("--dataset", type=str, required=True, # ['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100']
#                         help="Name of the dataset to use (e.g., MNIST, CIFAR10).")

#     parser.add_argument("--data_dir", type=str, default="./data",
#                         help="Directory to store/load datasets.")
#     parser.add_argument("--val_split", type=float, default=0.1,
#                         help="Fraction of data to use for validation set (0.0 to <1.0).")
#     parser.add_argument("--test_split", type=float, default=0.1, 
#                         help="Fraction of data to use for test set (0.0 to <1.0).")
#     parser.add_argument("--save_path", type=str, default="./trained_models",
#                         help="Directory to save trained models.")

#     parser.add_argument("--epochs", type=int, default=None,
#                         help="Number of training epochs (overrides schema default).")
#     parser.add_argument("--batch_size", type=int, default=None,
#                         help="Training batch size (overrides schema default).")
#     parser.add_argument("--lr", type=float, default=None,
#                         help="Learning rate (overrides schema default).")
#     optimizer_choices = [opt.value for opt in OptimizerType]
#     parser.add_argument("--optimizer", type=str, default=None, choices=optimizer_choices,
#                         help=f"Optimizer type (overrides schema default). Choose from: {', '.join(optimizer_choices)}")
#     parser.add_argument("--momentum", type=float, default=None,
#                         help="Momentum for SGD/RMSprop optimizer (overrides schema default).")
#     parser.add_argument("--weight_decay", type=float, default=None,
#                         help="Weight decay (L2 penalty) (overrides schema default).")

#     args = parser.parse_args()

#     if not (0 <= args.val_split < 1 and 0 <= args.test_split < 1 and (args.val_split + args.test_split) < 1):
#         print("Error: Invalid split fractions. val_split and test_split must be >= 0 and < 1, and their sum must be less than 1.")
#     else:
#         train_model(args)

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import time
import os

from tqdm import tqdm

from src.loading.data.loader import load_dataset
from src.loading.models.model_builder import create_model
from src.schema.training import OptimizerType, TrainingParams, Epoch, History, TrainingResult
from src.schema.dataset import Dataset

# from src.loading.models.alexnet import AlexNetArchitecture
# from src.loading.models.model_builder import load_model_architecture

def count_parameters(model: torch.nn.Module) -> int:
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n / 1e6

def get_optimizer(model: nn.Module, training_params: TrainingParams):
    
    kwargs = {}
    if training_params.learning_rate is not None:
        kwargs['lr'] = training_params.learning_rate
    if training_params.momentum is not None:
        kwargs['momentum'] = training_params.momentum
    if training_params.weight_decay is not None:
        kwargs['weight_decay'] = training_params.weight_decay
        
    if training_params.optimizer == OptimizerType.SGD:
        return optim.SGD(model.parameters(), **kwargs)
    elif training_params.optimizer == OptimizerType.ADAM:
        return optim.Adam(model.parameters(), **kwargs)
    elif training_params.optimizer == OptimizerType.RMSPROP:
        return optim.RMSprop(model.parameters(), **kwargs)
    elif training_params.optimizer == OptimizerType.ADGARAD:
        return optim.Adagrad(model.parameters(), **kwargs)

def evaluate_model(model: nn.Module, data_loader, criterion, device):
    """
    Evaluates the model on the provided data loader with a tqdm progress bar.

    Args:
        model (nn.Module): The model to evaluate.
        data_loader (DataLoader): DataLoader for the dataset to evaluate on.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to perform computation on.

    Returns:
        Tuple[float, float]: Average loss and accuracy in percentage.
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="Evaluating", leave=False)
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            avg_loss = total_loss / total if total > 0 else 0
            accuracy = 100 * correct / total if total > 0 else 0

            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{accuracy:.2f}%'
            })

    avg_loss = total_loss / total
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def train_model(model: nn.Module, dataset, training_params):
    """
    Trains the model on the provided dataset using the specified training parameters.

    Args:
        model (nn.Module): The model to train.
        dataset: An object with 'train_dataset' and 'test_dataset' attributes.
        training_params: An object with 'batch_size', 'epochs', and other training parameters.
    """
    print(f"Model parameters: {count_parameters(model):.3f} Million")
    
    torch.autograd.set_detect_anomaly(True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>>>>>>>>>>Using device: {device}")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, training_params)

    train_loader = DataLoader(dataset.train_dataset, batch_size=training_params.batch_size, shuffle=True)
    test_loader = DataLoader(dataset.test_dataset, batch_size=training_params.batch_size, shuffle=False)
    
    history = History(epochs=[])

    for epoch in range(training_params.epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{training_params.epochs}", leave=False)
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss = train_loss + loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total = train_total + labels.size(0)
            train_correct = train_correct + (predicted == labels).sum().item()

            progress_bar.set_postfix({
                'loss': f'{train_loss / train_total:.4f}',
                'acc': f'{100 * train_correct / train_total:.2f}%'
            })

        avg_train_loss = train_loss / train_total
        train_accuracy = 100 * train_correct / train_total
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Train Accuracy: {train_accuracy:.2f}%")

        test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
        print(f"Epoch {epoch+1} | Test Loss:  {test_loss:.4f} | Test Accuracy:  {test_accuracy:.2f}%")
        
        history.record_epoch(
            epoch=epoch+1,
            train_loss=avg_train_loss,
            test_loss=test_loss,
            train_accuracy=train_accuracy,
            test_accuracy=test_accuracy
        )
        
        print('-' * 20)

    torch.cuda.empty_cache()
    print("Finished Training")
    
    return TrainingResult(model, history)

# def train_model_with_args(args):
#     """
#     Trains a specified model on a specified dataset, using command-line args
#     to potentially override schema defaults.

#     Args:
#         args (argparse.Namespace): Parsed command-line arguments.
#     """
#     model_name = args.model
#     dataset_name = args.dataset
#     data_dir = args.data_dir
#     val_split = args.val_split
#     test_split = args.test_split
#     save_path_dir = args.save_path

#     print(f"Starting training process for Model: {model_name} on Dataset: {dataset_name}")

#     print("Loading dataset...")
#     try:
#         full_dataset, transform, in_channels, num_classes = load_dataset(dataset_name, data_dir=data_dir)
#         print(f"Dataset loaded: {len(full_dataset)} samples, In Channels: {in_channels}, Num Classes: {num_classes}")
#     except ValueError as e:
#         print(f"Error loading dataset: {e}")
#         return
#     except Exception as e:
#         print(f"An unexpected error occurred during dataset loading: {e}")
#         return

#     total_size = len(full_dataset)
#     test_size = int(test_split * total_size)
#     val_size = int(val_split * (total_size - test_size))
#     train_size = total_size - test_size - val_size

#     if train_size <= 0 or val_size <= 0 or test_size <= 0:
#         print(f"Error: Dataset split resulted in zero samples for one or more sets.")
#         print(f"Total: {total_size}, Train: {train_size}, Val: {val_size}, Test: {test_size}")
#         print("Adjust splits or check dataset size.")
#         return

#     try:
#         train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
#         print(f"Dataset split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
#     except Exception as e:
#         print(f"Error splitting dataset: {e}")
#         return

#     print("Loading model architecture schema...")
#     try:
#         architecture_schema = load_model_architecture(model_name)
#         schema_training_params: TrainingParams = architecture_schema.training
#         print(f"Schema defaults: Epochs={schema_training_params.epochs}, Batch={schema_training_params.batch_size}, LR={schema_training_params.learning_rate}, Optim={schema_training_params.optimizer.value}, Momentum={schema_training_params.momentum}, WeightDecay={schema_training_params.weight_decay}")
#     except ValueError as e:
#         print(f"Error loading model schema: {e}")
#         return
#     except Exception as e:
#         print(f"An unexpected error occurred loading model schema: {e}")
#         return

#     epochs = args.epochs if args.epochs is not None else schema_training_params.epochs
#     batch_size = args.batch_size if args.batch_size is not None else schema_training_params.batch_size
#     learning_rate = args.lr if args.lr is not None else schema_training_params.learning_rate
#     optimizer_str = args.optimizer if args.optimizer is not None else schema_training_params.optimizer.value
#     momentum = args.momentum if args.momentum is not None else schema_training_params.momentum
#     weight_decay = args.weight_decay if args.weight_decay is not None else schema_training_params.weight_decay

#     print(f"Effective Params: Epochs={epochs}, Batch={batch_size}, LR={learning_rate}, Optim={optimizer_str}, Momentum={momentum}, WeightDecay={weight_decay}")

#     print("Building model...")
#     try:
#         model = create_model(architecture_schema, in_channels=in_channels, num_classes=num_classes)
#     except Exception as e:
#         print(f"Error building model from schema: {e}")
#         return

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     print(f"Model built and moved to device: {device}")

#     print("Attempting to initialize lazy layers (if any)...")
#     try:
#         sample_img, _ = full_dataset[0]
#         dummy_batch_size = 2 
#         dummy_input = sample_img.unsqueeze(0).repeat(dummy_batch_size, 1, 1, 1).to(device)

#         model.eval() 
#         with torch.no_grad():
#             _ = model(dummy_input)
#         model.train() 
#         print("Lazy layers initialized successfully.")
#     except Exception as e:
#         print(f"Warning: Error during lazy layer initialization: {e}")
#         print("Proceeding, but this might cause issues if model has lazy layers.")

#     criterion = nn.CrossEntropyLoss()
#     try:
#         optimizer = get_optimizer(
#             optimizer_str,
#             model.parameters(),
#             lr=learning_rate,
#             momentum=momentum,
#             weight_decay=weight_decay
#         )
#     except ValueError as e:
#         print(f"Error creating optimizer: {e}")
#         return

#     num_workers = 2 if device.type == 'cuda' else 0 
#     try:
#         train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False)
#         val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False)
#         test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False)
#         print("Optimizer, Loss, and DataLoaders configured.")
#     except Exception as e:
#         print(f"Error creating DataLoaders: {e}")
#         return

#     print(f"Starting training loop for {epochs} epochs...")
#     start_time = time.time()
#     best_val_accuracy = 0.0
#     best_model_path = ""

#     for epoch in range(epochs):
#         epoch_start_time = time.time()
#         model.train() 
#         running_loss = 0.0
#         correct_train = 0
#         total_train = 0

#         for i, data in enumerate(train_loader):
#             try:
#                 inputs, labels = data[0].to(device), data[1].to(device)

#                 optimizer.zero_grad()

#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)
#                 loss.backward()
#                 optimizer.step()

#                 running_loss += loss.item()
#                 _, predicted = torch.max(outputs.data, 1)
#                 total_train += labels.size(0)
#                 correct_train += (predicted == labels).sum().item()

#                 if (i + 1) % 100 == 0:
#                     print(f'  Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

#             except Exception as batch_error:
#                 print(f"Error during training batch {i+1} in epoch {epoch+1}: {batch_error}")
#                 continue 

#         epoch_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0
#         epoch_acc_train = (100 * correct_train / total_train) if total_train > 0 else 0

#         model.eval() 
#         correct_val = 0
#         total_val = 0
#         val_loss = 0.0
#         with torch.no_grad():
#             for data in val_loader:
#                 try:
#                     images, labels = data[0].to(device), data[1].to(device)
#                     outputs = model(images)
#                     loss = criterion(outputs, labels)
#                     val_loss += loss.item()
#                     _, predicted = torch.max(outputs.data, 1)
#                     total_val += labels.size(0)
#                     correct_val += (predicted == labels).sum().item()
#                 except Exception as val_batch_error:
#                     print(f"Error during validation batch: {val_batch_error}")
#                     continue 

#         epoch_acc_val = (100 * correct_val / total_val) if total_val > 0 else 0
#         epoch_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
#         epoch_end_time = time.time()

#         print(f"Epoch {epoch+1}/{epochs} Summary: ")
#         print(f"  Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc_train:.2f}%")
#         print(f"  Val Loss:   {epoch_val_loss:.4f}, Val Acc:   {epoch_acc_val:.2f}%")
#         print(f"  Epoch Time: {epoch_end_time - epoch_start_time:.2f} seconds")

#         if epoch_acc_val > best_val_accuracy:
#             best_val_accuracy = epoch_acc_val
#             os.makedirs(save_path_dir, exist_ok=True) 
#             current_best_path = os.path.join(save_path_dir, f'{model_name}_{dataset_name}_best.pth')
#             try:
#                 torch.save(model.state_dict(), current_best_path)
#                 best_model_path = current_best_path 
#                 print(f"  Best validation accuracy improved to {best_val_accuracy:.2f}%. Model saved to {best_model_path}")
#             except Exception as save_error:
#                 print(f"Error saving model: {save_error}")

#     total_training_time = time.time() - start_time
#     print(f"Finished Training. Total time: {total_training_time // 60:.0f}m {total_training_time % 60:.0f}s")

#     print("\nLoading best model for final test evaluation...")
#     if os.path.exists(best_model_path):
#         print(f"Loading model from: {best_model_path}")
#         try:
#             final_model = create_model(architecture_schema, in_channels=in_channels, num_classes=num_classes)
#             final_model.load_state_dict(torch.load(best_model_path, map_location=device)) 
#             final_model.to(device)
#             final_model.eval()

#             correct_test = 0
#             total_test = 0
#             with torch.no_grad():
#                 for data in test_loader:
#                     try:
#                         images, labels = data[0].to(device), data[1].to(device)
#                         outputs = final_model(images)
#                         _, predicted = torch.max(outputs.data, 1)
#                         total_test += labels.size(0)
#                         correct_test += (predicted == labels).sum().item()
#                     except Exception as test_batch_error:
#                          print(f"Error during testing batch: {test_batch_error}")
#                          continue 

#             test_accuracy = (100 * correct_test / total_test) if total_test > 0 else 0
#             print(f"Final Test Accuracy on {len(test_dataset)} samples: {test_accuracy:.2f}%")
#         except Exception as load_error:
#             print(f"Error loading or evaluating best model: {load_error}")
#     else:
#         print(f"Could not find best model at expected path: {best_model_path}. Skipping final test evaluation.")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Train a CNN model defined by schema on a specified dataset.")

#     parser.add_argument("--model", type=str, required=True, # ['AlexNet', 'VGG', 'ResNet'],
#                         help="Name of the model architecture to train (e.g., AlexNet, VGG, ResNet).")
#     parser.add_argument("--dataset", type=str, required=True, # ['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100']
#                         help="Name of the dataset to use (e.g., MNIST, CIFAR10).")

#     parser.add_argument("--data_dir", type=str, default="./data",
#                         help="Directory to store/load datasets.")
#     parser.add_argument("--val_split", type=float, default=0.1,
#                         help="Fraction of data to use for validation set (0.0 to <1.0).")
#     parser.add_argument("--test_split", type=float, default=0.1, 
#                         help="Fraction of data to use for test set (0.0 to <1.0).")
#     parser.add_argument("--save_path", type=str, default="./trained_models",
#                         help="Directory to save trained models.")

#     parser.add_argument("--epochs", type=int, default=None,
#                         help="Number of training epochs (overrides schema default).")
#     parser.add_argument("--batch_size", type=int, default=None,
#                         help="Training batch size (overrides schema default).")
#     parser.add_argument("--lr", type=float, default=None,
#                         help="Learning rate (overrides schema default).")
#     optimizer_choices = [opt.value for opt in OptimizerType]
#     parser.add_argument("--optimizer", type=str, default=None, choices=optimizer_choices,
#                         help=f"Optimizer type (overrides schema default). Choose from: {', '.join(optimizer_choices)}")
#     parser.add_argument("--momentum", type=float, default=None,
#                         help="Momentum for SGD/RMSprop optimizer (overrides schema default).")
#     parser.add_argument("--weight_decay", type=float, default=None,
#                         help="Weight decay (L2 penalty) (overrides schema default).")

#     args = parser.parse_args()

#     if not (0 <= args.val_split < 1 and 0 <= args.test_split < 1 and (args.val_split + args.test_split) < 1):
#         print("Error: Invalid split fractions. val_split and test_split must be >= 0 and < 1, and their sum must be less than 1.")
#     else:
#         train_model(args)
