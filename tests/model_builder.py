import torch

from src.loading.models.alexnet import AlexNetArchitecture
from src.loading.models.model_builder import create_model

test = 'AlexNet'

if test == 'AlexNet':
    model_name = 'AlexNet'
    model = create_model(AlexNetArchitecture, in_channels=3, num_classes=10)
    print(model)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(f"Output shape: {y.shape}")