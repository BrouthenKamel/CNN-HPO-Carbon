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
    def __init__(self, reference: MobileNetHP = original_hp, num_blocks: int = 11, freeze_blocks_until: int = 0):
        self.reference = reference
        self.num_blocks = num_blocks
        self.freeze_blocks_until = freeze_blocks_until
        
        self.activation_choices = ["ReLU", "Hardswish"]
        self.squeeze_factor_choices = [4, 8]
        self.se_activation_choices = ["Hardsigmoid", "Sigmoid"]

        self.channel_choices = multiples_of(8, 1, 20)
        self.expand_channels_choices = multiples_of(8, 1, 40)
        self.classifier_neuron_choices = multiples_of(256, 1, 4)

        self.kernel_size_choices = [3, 5, 7]
        self.stride_choices = [1, 2]
        self.dropout_choices = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        
        self.last_conv_upsample_choices = [2, 4, 6, 8]

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

    def neighbor(
        self,
        reference_hp: MobileNetHP,
        block_modification_ratio: float = 0.5,
        param_modification_ratio: float = 0.2,
        perturbation_intensity: int = 1,
        perturbation_strategy: str = "local",
    ) -> MobileNetHP:

        def perturb_choice(choices, current, intensity, strategy):
            max_choices = max(choices)
            min_choices = min(choices)
            if current > max_choices:
                current = max_choices
            if current < min_choices:
                current = min_choices
            idx = choices.index(current)
            if strategy == "random":
                return random.choice(choices)
            elif strategy == "local":
                direction = random.choice([-1, 1])
                new_idx = max(0, min(len(choices) - 1, idx + direction * intensity))
                return choices[new_idx]
            return current

        def should_modify(ratio):
            decision = random.random() < ratio
            return decision

        # Perturb Initial Conv
        ref_ic = reference_hp.initial_conv_hp
        if self.freeze_blocks_until == 0 and should_modify(block_modification_ratio):
            initial_conv_hp = ConvBNActivationHP(
                channels=perturb_choice(self.channel_choices, ref_ic.channels, perturbation_intensity, perturbation_strategy) if should_modify(param_modification_ratio) else ref_ic.channels,
                kernel_size=perturb_choice(self.kernel_size_choices, ref_ic.kernel_size, perturbation_intensity, perturbation_strategy) if should_modify(param_modification_ratio) else ref_ic.kernel_size,
                stride=perturb_choice(self.stride_choices, ref_ic.stride, perturbation_intensity, perturbation_strategy) if should_modify(param_modification_ratio) else ref_ic.stride,
                activation=perturb_choice(self.activation_choices, ref_ic.activation, perturbation_intensity, perturbation_strategy) if should_modify(param_modification_ratio) else ref_ic.activation
            )
        else:
            initial_conv_hp = ref_ic

        # Perturb IR Blocks
        inverted_residual_hps = []
        prev_out_ch = initial_conv_hp.channels
        for i, ref_ir in enumerate(reference_hp.inverted_residual_hps):
            ref_conv = ref_ir.conv_bn_activation_hp

            if ((i+1) >= self.freeze_blocks_until) and should_modify(block_modification_ratio):
                out_ch = perturb_choice(
                    [c for c in self.channel_choices if c >= prev_out_ch],
                    ref_conv.channels,
                    perturbation_intensity,
                    perturbation_strategy
                ) if should_modify(param_modification_ratio) else ref_conv.channels

                expand_ch = perturb_choice(
                    [c for c in self.expand_channels_choices if c >= out_ch],
                    ref_ir.expanded_channels,
                    perturbation_intensity,
                    perturbation_strategy
                ) if should_modify(param_modification_ratio) else ref_ir.expanded_channels

                kernel_size = perturb_choice(self.kernel_size_choices, ref_conv.kernel_size, perturbation_intensity, perturbation_strategy) if should_modify(param_modification_ratio) else ref_conv.kernel_size
                stride = perturb_choice(self.stride_choices, ref_conv.stride, perturbation_intensity, perturbation_strategy) if should_modify(param_modification_ratio) else ref_conv.stride
                activation = perturb_choice(self.activation_choices, ref_conv.activation, perturbation_intensity, perturbation_strategy) if should_modify(param_modification_ratio) else ref_conv.activation

                se_hp = ref_ir.se_hp
                if ref_ir.use_se and should_modify(param_modification_ratio):
                    se_hp = SqueezeExcitationHP(
                        squeeze_factor=perturb_choice(self.squeeze_factor_choices, se_hp.squeeze_factor, perturbation_intensity, perturbation_strategy),
                        activation=perturb_choice(self.se_activation_choices, se_hp.activation, perturbation_intensity, perturbation_strategy)
                    )
            else:
                out_ch = ref_conv.channels
                expand_ch = ref_ir.expanded_channels
                kernel_size = ref_conv.kernel_size
                stride = ref_conv.stride
                activation = ref_conv.activation
                se_hp = ref_ir.se_hp

            inverted_residual_hps.append(InvertedResidualHP(
                expand_channels=expand_ch,
                use_se=ref_ir.use_se,
                se_hp=se_hp,
                conv_bn_activation_hp=ConvBNActivationHP(
                    channels=out_ch,
                    kernel_size=kernel_size,
                    stride=stride,
                    activation=activation
                )
            ))

            prev_out_ch = out_ch

        # Perturb Last Conv
        if self.freeze_blocks_until != 'all' and should_modify(block_modification_ratio):
            last_conv_upsample = perturb_choice(
                self.last_conv_upsample_choices,
                reference_hp.last_conv_upsample,
                perturbation_intensity,
                perturbation_strategy
            ) if should_modify(param_modification_ratio) else reference_hp.last_conv_upsample

            last_conv_hp = ConvBNActivationHP(
                channels=prev_out_ch * last_conv_upsample,
                kernel_size=1,
                stride=1,
                activation=perturb_choice(self.activation_choices, reference_hp.last_conv_hp.activation, perturbation_intensity, perturbation_strategy) if should_modify(param_modification_ratio) else reference_hp.last_conv_hp.activation
            )
        else:
            last_conv_upsample = reference_hp.last_conv_upsample
            last_conv_hp = reference_hp.last_conv_hp

        # Perturb Classifier
        ref_cls = reference_hp.classifier_hp
        classifier_hp = ClassifierHP(
            neurons=perturb_choice(self.classifier_neuron_choices, ref_cls.neurons, perturbation_intensity, perturbation_strategy) if should_modify(param_modification_ratio) else ref_cls.neurons,
            activation=perturb_choice(self.activation_choices, ref_cls.activation, perturbation_intensity, perturbation_strategy) if should_modify(param_modification_ratio) else ref_cls.activation,
            dropout_rate=perturb_choice(self.dropout_choices, ref_cls.dropout_rate, perturbation_intensity, perturbation_strategy) if should_modify(param_modification_ratio) else ref_cls.dropout_rate
        )

        return MobileNetHP(
            initial_conv_hp=initial_conv_hp,
            inverted_residual_hps=inverted_residual_hps,
            last_conv_upsample=last_conv_upsample,
            last_conv_hp=last_conv_hp,
            classifier_hp=classifier_hp
        )

if __name__ == "__main__":
    hp_space = MobileNetHPSpace()
    sampled_hp = hp_space.sample()
    print(sampled_hp.to_dict())
    neighbor_hp = hp_space.neighbor(sampled_hp)
    print(neighbor_hp.to_dict())
