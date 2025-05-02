import time

from src.loading.data.loader import load_dataset
from src.schema.dataset import DatasetName

from src.training.train import train_model, count_parameters

from src.schema.training import TrainingParams, OptimizerType

from src.loading.models.mobilenet.space import MobileNetHPSpace
from src.loading.models.mobilenet.config import MobileNetConfig
from src.loading.models.mobilenet.model import MobileNetV3Small

training_params = TrainingParams(
    epochs = 1,
    batch_size = 64,
    learning_rate = 0.001,
    optimizer = OptimizerType.ADAM,
    momentum = None,
    weight_decay = None,
)

n = 1

record = []

dataset_name = DatasetName.CIFAR10

dataset = load_dataset(dataset_name)

hp_space = MobileNetHPSpace(num_blocks=11)

for i in range(n):
    
    print()
    
    print("Sampling MobilerNet Architecture...")
    hp = hp_space.sample()
    config = MobileNetConfig.from_hp(hp)
    
    print("Sampled Architecture:")
    
    model = MobileNetV3Small(config, dataset.num_classes)
    print(f"Model Instantiated successfully! Parameters: {count_parameters(model)} Million")
    print(model)
            
    s = time.time()
    res = train_model(model, dataset, training_params)    
    e = time.time()
    
    print(f"Training time: {(e - s) / 60:.2f} minutes")
    
    record.append({
        'model_state_dict': res.model.state_dict(),
        'history': res.history,
    })
    
    print ("=" * 20)
    
print("Training completed!")
print(record)
