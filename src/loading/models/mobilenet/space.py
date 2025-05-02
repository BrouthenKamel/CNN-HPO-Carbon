import random

from src.loading.models.mobilenet.hp import (
    MobileNetHP,
    ConvBNActivationHP,
    InvertedResidualHP,
    SqueezeExcitationHP,
    ClassifierHP, 
    original_hp
)

def multiples_of(base: int, start: int, stop: int) -> list[int]:
    return [base * i for i in range(start, stop+1)]

class MobileNetHPSpace:
    def __init__(self, reference: MobileNetHP = original_hp, num_blocks=11):
        self.reference = reference
        
        self.activation_choices = ["ReLU", "Hardswish"]
        self.squeeze_factor_choices = [2, 4, 8]
        self.se_activation_choices = ["Hardsigmoid", "Sigmoid"]

        self.channel_choices = multiples_of(8, 1, 20)
        self.expand_channels_choices = multiples_of(8, 1, 40)
        self.classifier_neuron_choices = multiples_of(256, 1, 4)

        self.kernel_size_choices = [3, 5, 7]
        self.stride_choices = [1, 2]
        self.dropout_choices = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        
        self.last_conv_upsample_choices = [2, 4, 6, 8]

        self.num_blocks = num_blocks

    def sample(self) -> MobileNetHP:
        def random_conv_hp(channels=None, max_channels=None):
            return ConvBNActivationHP(
                channels=channels if channels else random.choice([c for c in self.channel_choices if c <= max_channels]),
                kernel_size=random.choice(self.kernel_size_choices),
                stride=random.choice(self.stride_choices),
                activation=random.choice(self.activation_choices)
            )

        def random_se_hp():
            return SqueezeExcitationHP(
                squeeze_factor=random.choice(self.squeeze_factor_choices),
                activation=random.choice(self.se_activation_choices)
            )

        def random_ir_hp(prev_out_ch, reference_ir: InvertedResidualHP):
            
            ref_conv = reference_ir.conv_bn_activation_hp
            out_ch = random.choice([c for c in self.channel_choices if c <= ref_conv.channels and c >= prev_out_ch])
            expand_ch = random.choice([c for c in self.expand_channels_choices if c <= reference_ir.expanded_channels and c >= out_ch])
            use_se = reference_ir.use_se
            se_hp = random_se_hp() if use_se else None
            
            return InvertedResidualHP(
                expand_channels=expand_ch,
                use_se=use_se,
                se_hp=se_hp,
                conv_bn_activation_hp=ConvBNActivationHP(
                    channels=out_ch,
                    kernel_size=random.choice(self.kernel_size_choices),
                    stride=random.choice(self.stride_choices),
                    activation=random.choice(self.activation_choices)
                )
            ), out_ch

        ref_ic = self.reference.initial_conv_hp

        initial_channels = random.choice([c for c in self.channel_choices if c <= ref_ic.channels])
        initial_conv_hp = random_conv_hp(channels=initial_channels)

        inverted_residual_hps = []
        prev_out_ch = initial_channels
        for ref_ir in self.reference.inverted_residual_hps:
            ir_hp, prev_out_ch = random_ir_hp(prev_out_ch, ref_ir)
            inverted_residual_hps.append(ir_hp)

        ref_lc = self.reference.last_conv_hp
        last_conv_upsample = random.choice(self.last_conv_upsample_choices)
        last_conv_hp = ConvBNActivationHP(
            channels=prev_out_ch * last_conv_upsample,
            kernel_size=1,
            stride=1,
            activation=random.choice(self.activation_choices)
        )

        classifier_hp = ClassifierHP(
            neurons=random.choice(self.classifier_neuron_choices),
            activation=random.choice(self.activation_choices),
            dropout_rate=random.choice(self.dropout_choices)
        )

        return MobileNetHP(
            initial_conv_hp=initial_conv_hp,
            inverted_residual_hps=inverted_residual_hps,
            last_conv_upsample=random.choice(self.last_conv_upsample_choices),
            last_conv_hp=last_conv_hp,
            classifier_hp=classifier_hp
        )
