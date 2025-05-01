from src.loading.models.mobilenet.utils import make_divisible

class SqueezeExcitationHP:
    def __init__(self, squeeze_factor: int, activation: str):
        self.squeeze_factor = squeeze_factor
        self.activation = activation
        
class ConvBNActivationHP:
    def __init__(self, kernel_size: int, stride: int, padding: int, activation: str, channels: int = None):
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation = activation
        
class InvertedResidualHP:
    def __init__(self, expand_channels: int, use_se: bool, se_hp: SqueezeExcitationHP, conv_bn_activation_hp: ConvBNActivationHP):
        self.expanded_channels = make_divisible(expand_channels, 8)
        self.use_se = use_se
        self.se_hp = se_hp
        self.conv_bn_activation_hp = conv_bn_activation_hp
        
class ClassifierHP:
    def __init__(self, neurons: int, activation: str, dropout_rate: float):
        self.neurons = neurons
        self.activation = activation
        self.dropout_rate = dropout_rate

class MobileNetHP:
    def __init__(self, initial_conv_hp: ConvBNActivationHP, inverted_residual_hps: list[InvertedResidualHP], last_conv_upsample: int, last_conv_hp: ConvBNActivationHP, classifier_hp: ClassifierHP):
        self.initial_conv_hp = initial_conv_hp
        self.inverted_residual_hps = inverted_residual_hps
        self.last_conv_upsample = last_conv_upsample
        self.last_conv_hp = last_conv_hp
        self.classifier_hp = classifier_hp
        
hp = MobileNetHP(
    initial_conv_hp=ConvBNActivationHP(channels=16, kernel_size=3, stride=2, padding=1, activation="Hardswish"),
    inverted_residual_hps=[
        InvertedResidualHP(
            expand_channels=16,
            use_se=True,
            se_hp=SqueezeExcitationHP(squeeze_factor=4, activation="Hardsigmoid"),
            conv_bn_activation_hp=ConvBNActivationHP(channels=16, kernel_size=3, stride=2, padding=1, activation="ReLU"),
        ),
        InvertedResidualHP(
            expand_channels=72,
            use_se=False,
            se_hp=None,
            conv_bn_activation_hp=ConvBNActivationHP(channels=24, kernel_size=3, stride=2, padding=1, activation="ReLU"),
        ),
        InvertedResidualHP(
            expand_channels=88,
            use_se=False,
            se_hp=None,
            conv_bn_activation_hp=ConvBNActivationHP(channels=24, kernel_size=3, stride=1, padding=1, activation="Hardswish"),
        ),
        InvertedResidualHP(
            expand_channels=96,
            use_se=True,
            se_hp=SqueezeExcitationHP(squeeze_factor=4, activation="Hardsigmoid"),
            conv_bn_activation_hp=ConvBNActivationHP(channels=40, kernel_size=5, stride=2, padding=2, activation="Hardswish"),
        ),
        InvertedResidualHP(
            expand_channels=240,
            use_se=True,
            se_hp=SqueezeExcitationHP(squeeze_factor=4, activation="Hardsigmoid"),
            conv_bn_activation_hp=ConvBNActivationHP(channels=40, kernel_size=5, stride=1, padding=2, activation="Hardswish"),
        ),
        InvertedResidualHP(
            expand_channels=240,
            use_se=True,
            se_hp=SqueezeExcitationHP(squeeze_factor=4, activation="Hardsigmoid"),
            conv_bn_activation_hp=ConvBNActivationHP(channels=40, kernel_size=5, stride=1, padding=2, activation="Hardswish"),
        ),
        InvertedResidualHP(
            expand_channels=120,
            use_se=True,
            se_hp=SqueezeExcitationHP(squeeze_factor=4, activation="Hardsigmoid"),
            conv_bn_activation_hp=ConvBNActivationHP(channels=48, kernel_size=5, stride=1, padding=2, activation="Hardswish"),
        ),
        InvertedResidualHP(
            expand_channels=144,
            use_se=True,
            se_hp=SqueezeExcitationHP(squeeze_factor=4, activation="Hardsigmoid"),
            conv_bn_activation_hp=ConvBNActivationHP(channels=48, kernel_size=5, stride=1, padding=2, activation="Hardswish"),
        ),
        InvertedResidualHP(
            expand_channels=288,
            use_se=True,
            se_hp=SqueezeExcitationHP(squeeze_factor=4, activation="Hardsigmoid"),
            conv_bn_activation_hp=ConvBNActivationHP(channels=96, kernel_size=5, stride=1, padding=2, activation="Hardswish"),
        ),
        InvertedResidualHP(
            expand_channels=576,
            use_se=True,
            se_hp=SqueezeExcitationHP(squeeze_factor=4, activation="Hardsigmoid"),
            conv_bn_activation_hp=ConvBNActivationHP(channels=96, kernel_size=5, stride=1, padding=2, activation="Hardswish"),
        ),
        InvertedResidualHP(
            expand_channels=576,
            use_se=True,
            se_hp=SqueezeExcitationHP(squeeze_factor=4, activation="Hardsigmoid"),
            conv_bn_activation_hp=ConvBNActivationHP(channels=96, kernel_size=5, stride=1, padding=2, activation="Hardswish"),
        ),
    ],
    last_conv_upsample=6,
    last_conv_hp=ConvBNActivationHP(kernel_size=1, stride=1, padding=0, activation="Hardswish"),
    classifier_hp=ClassifierHP(neurons=1024, activation="Hardswish", dropout_rate=0.2)
)
