import torch
import torchvision.models as models

from src.schema.model import ModelFamily

# Define a mapping from family to model loading function
family_to_model = {
    ModelFamily.VGG: lambda weights: models.vgg11_bn(weights=weights),
    ModelFamily.RESNET: lambda weights: models.resnet18(weights=weights),
    ModelFamily.EFFICIENTNET: lambda weights: models.efficientnet_b0(weights=weights),
    ModelFamily.MOBILENET: lambda weights: models.mobilenet_v3_small(weights=weights),
    ModelFamily.DENSENET: lambda weights: models.densenet121(weights=weights),
    ModelFamily.REGNET: lambda weights: models.regnet_y_400mf(weights=weights),
    ModelFamily.SQUEEZENET: lambda weights: models.squeezenet1_0(weights=weights),
}

default_weights = {
    ModelFamily.VGG: models.VGG11_BN_Weights.IMAGENET1K_V1,
    ModelFamily.RESNET: models.ResNet18_Weights.IMAGENET1K_V1,
    ModelFamily.EFFICIENTNET: models.EfficientNet_B0_Weights.IMAGENET1K_V1,
    ModelFamily.MOBILENET: models.MobileNet_V3_Small_Weights.IMAGENET1K_V1,
    ModelFamily.DENSENET: models.DenseNet121_Weights.IMAGENET1K_V1,
    ModelFamily.REGNET: models.RegNet_Y_400MF_Weights.IMAGENET1K_V1,
    ModelFamily.SQUEEZENET: models.SqueezeNet1_0_Weights.IMAGENET1K_V1,
}

def load_pretrained_model(family: ModelFamily, pretrained: bool = True):
    try:
        if pretrained:
            weights = default_weights[family]
        else:
            weights = None
        
        model = family_to_model[family](weights)
        return model

    except KeyError:
        raise ValueError(f"Unknown model family: {family}")

