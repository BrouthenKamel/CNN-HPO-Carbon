import time

from src.loading.data.loader import load_dataset
from src.schema.dataset import DatasetName

from src.loading.models.pretrained import load_pretrained_model, suggest_output_layer
from src.training.train import train_model, count_parameters

from src.schema.model import ModelFamily
from src.schema.training import TrainingParams, OptimizerType

training_params = TrainingParams(
    epochs = 1,
    batch_size = 128,
    learning_rate = 0.001,
    optimizer = OptimizerType.ADAM,
    momentum = None,
    weight_decay = None
)

# dataset_epochs = {
#     'pretrained': {
#         DatasetName.MNIST: 5,
#         DatasetName.FashionMNIST: 5,
#         DatasetName.CIFAR10: 5,
#         DatasetName.CIFAR100: 5,
#     },
#     'not_pretrained': {
#         DatasetName.MNIST: 5,
#         DatasetName.FashionMNIST: 5,
#         DatasetName.CIFAR10: 5,
#         DatasetName.CIFAR100: 5,
#     }
# }
    
# for dataset_name in [DatasetName.MNIST, DatasetName.FashionMNIST, DatasetName.CIFAR10]:
for dataset_name in [DatasetName.CIFAR10, DatasetName.CIFAR100]:

    dataset = load_dataset(dataset_name)
    
    for family in [
        # ModelFamily.VGG,
        # ModelFamily.RESNET,
        # ModelFamily.EFFICIENTNET,
        # ModelFamily.MOBILENET,
        # ModelFamily.DENSENET,
        # ModelFamily.REGNET,
        # ModelFamily.SQUEEZENET,
        ModelFamily.MOBILENETV3,
    ]:
        
        for pretrained in [False, True]:
            
            print()
            
            print(f"Training ({family.value}) on ({dataset_name.value}) with pretrained={pretrained}")
            
            model = load_pretrained_model(family, dataset.num_classes, pretrained)
            
            s = time.time()
            
            train_model(model, dataset, training_params)
            
            print(f"Parameters: {count_parameters(model)} Million")
            
            # suggest_output_layer(model, dataset.num_classes)
            
            e = time.time()
            
            print(f"Training time: {(e - s) / 60:.2f} minutes")
            
            print ("=" * 20)
