import time

from src.loading.data.loader import load_dataset
from src.schema.dataset import DatasetName

from src.loading.models.pretrained import load_pretrained_model
from src.training.train import train_model

from src.schema.model import ModelFamily
from src.schema.training import TrainingParams, OptimizerType

training_params = TrainingParams(
    epochs = 1,
    batch_size = 64,
    learning_rate = 0.001,
    optimizer = OptimizerType.ADAM,
    momentum = None,
    weight_decay = None
)
    
# for dataset_name in [DatasetName.CIFAR10, DatasetName.CIFAR100]:
for dataset_name in [DatasetName.CIFAR10]:

    dataset = load_dataset(dataset_name)
    
    for family in [
        ModelFamily.RESNET,
        ModelFamily.EFFICIENTNET,
        ModelFamily.MOBILENET,
        ModelFamily.DENSENET,
        ModelFamily.REGNET,
        ModelFamily.SQUEEZENET,
        ModelFamily.VGG,
    ]:
        
        for pretrained in [True, False]:
            
            print()
            
            print(f"Training ({family.value}) on ({dataset_name.value}) with pretrained={pretrained}")
            
            model = load_pretrained_model(family, pretrained)
            
            s = time.time()
            
            train_model(model, dataset, training_params)
            
            e = time.time()
            
            print(f"Training time: {(e - s) / 60:.2f} minutes")
            
            print ("=" * 20)
