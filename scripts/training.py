

from src.loading.data.loader import load_dataset
from src.schema.dataset import DatasetName

from src.loading.models.pretrained import load_pretrained_model
from src.training.train import train_model

from src.schema.model import ModelFamily
from src.schema.training import TrainingParams, OptimizerType

training_params = TrainingParams(
    epochs = 10,
    batch_size = 32,
    learning_rate = 0.001,
    optimizer = OptimizerType.ADAM,
    momentum = None,
    weight_decay = None
)
    
for dataset_name in [DatasetName.CIFAR10, DatasetName.CIFAR100]:
    
    dataset = load_dataset(dataset_name)
    
    for family in [
        ModelFamily.VGG,
        ModelFamily.RESNET,
        ModelFamily.EFFICIENTNET,
        ModelFamily.MOBILENET,
        ModelFamily.DENSENET,
        ModelFamily.REGNET,
        ModelFamily.SQUEEZENET
    ]:
        
        for pretrained in [True, False]:
            
            print(f"Training {family} on {dataset_name} with pretrained={pretrained}")
            
            model = load_pretrained_model(family, pretrained)
            
            train_model(model, dataset, training_params)
