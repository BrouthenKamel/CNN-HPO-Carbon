import random
from typing import List, Optional

from src.loading.models.resnet18.hp import (
    ConvLayerHP,
    BasicBlockHP,
    ClassifierHP,
    ResNetHP,
    POSSIBLE_ACTIVATIONS, 
)


KERNEL_SIZE_CHOICES = [1, 3, 5, 7, 9]
STRIDE_CHOICES = [1, 2, 3]
FILTER_CHOICES_INITIAL = [16, 32, 48, 64, 96, 128]
FILTER_CHOICES_STAGE = [32, 64, 96, 128, 192, 256, 320, 384, 448, 512, 640, 768]
BLOCK_COUNT_CHOICES = [1, 2, 3, 4, 6, 8]
MLP_NEURON_CHOICES = [64, 128, 256, 384, 512, 768, 1024, 1536, 2048]
NUM_MLP_LAYERS_CHOICES = [0, 1, 2, 3]
DROPOUT_CHOICES = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
BOOL_CHOICES = [True, False]
NUM_STAGES = [2, 3, 4, 5, 6]

class ResNetHPSpace:
    """Defines the search space for ResNet hyperparameters."""

    def __init__(self,
                 activation_choices: List[str] = POSSIBLE_ACTIVATIONS,
                 kernel_size_choices: List[int] = KERNEL_SIZE_CHOICES,
                 stride_choices: List[int] = STRIDE_CHOICES,
                 filter_choices_initial: List[int] = FILTER_CHOICES_INITIAL,
                 filter_choices_stage: List[int] = FILTER_CHOICES_STAGE,
                 block_count_choices: List[int] = BLOCK_COUNT_CHOICES,
                 mlp_neuron_choices: List[int] = MLP_NEURON_CHOICES,
                 num_mlp_layers_choices: List[int] = NUM_MLP_LAYERS_CHOICES,
                 dropout_choices: List[float] = DROPOUT_CHOICES,
                 initial_maxpool_choices: List[bool] = BOOL_CHOICES,
                 num_stages: int = NUM_STAGES
                 ):
        self.activation_choices = activation_choices
        self.kernel_size_choices = kernel_size_choices
        self.stride_choices = stride_choices 
        self.filter_choices_initial = filter_choices_initial
        self.filter_choices_stage = filter_choices_stage
        self.block_count_choices = block_count_choices
        self.mlp_neuron_choices = mlp_neuron_choices
        self.num_mlp_layers_choices = num_mlp_layers_choices
        self.dropout_choices = dropout_choices
        self.initial_maxpool_choices = initial_maxpool_choices
        self.num_stages = num_stages

    def sample(self) -> ResNetHP:
        """Randomly samples a ResNetHP configuration from the defined space."""

        initial_conv_hp = ConvLayerHP(
            filters=random.choice(self.filter_choices_initial),
            kernel_size=random.choice(self.kernel_size_choices),
            stride=random.choice([1, 2]), 
            activation=random.choice(self.activation_choices)
        )
        initial_maxpool = random.choice(self.initial_maxpool_choices)

        stage_block_counts = []
        stage_block_hps = []
        current_stride_stage = 1 
        for i in range(random.choice(self.num_stages)):
            block_count = random.choice(self.block_count_choices)
            stage_block_counts.append(block_count)

            stage_stride = 1 if i == 0 else 2
            # stage_stride = random.choice(self.stride_choices)

            stage_hp = BasicBlockHP(
                filters=random.choice(self.filter_choices_stage),
                stride=stage_stride,
                activation=random.choice(self.activation_choices)
            )
            stage_block_hps.append(stage_hp)

        # Classifier
        num_mlp_layers = random.choice(self.num_mlp_layers_choices)
        mlp_neurons = [random.choice(self.mlp_neuron_choices) for _ in range(num_mlp_layers)]
        classifier_hp = ClassifierHP(
            neurons=mlp_neurons,
            activation=random.choice(self.activation_choices),
            dropout_rate=random.choice(self.dropout_choices)
        )

        return ResNetHP(
            initial_conv_hp=initial_conv_hp,
            initial_maxpool=initial_maxpool,
            stage_block_counts=stage_block_counts,
            stage_block_hps=stage_block_hps,
            classifier_hp=classifier_hp
        )

    def neighbor(self, reference_hp: ResNetHP, modification_prob: float = 0.3) -> ResNetHP:
        """
        Generates a neighbor configuration by making small random modifications
        to a reference ResNetHP.
        """
        new_initial_conv_hp = ConvLayerHP(**reference_hp.initial_conv_hp.to_dict())
        new_stage_block_counts = list(reference_hp.stage_block_counts)
        new_stage_block_hps = [BasicBlockHP(**hp.to_dict()) for hp in reference_hp.stage_block_hps]
        new_classifier_hp = ClassifierHP(**reference_hp.classifier_hp.to_dict())
        new_initial_maxpool = reference_hp.initial_maxpool

        if random.random() < modification_prob:
            new_initial_conv_hp.filters = random.choice(self.filter_choices_initial)
        if random.random() < modification_prob:
            new_initial_conv_hp.kernel_size = random.choice(self.kernel_size_choices)
        # if random.random() < modification_prob:
        #     new_initial_conv_hp.stride = random.choice([1, 2])
        if random.random() < modification_prob:
            new_initial_conv_hp.activation = random.choice(self.activation_choices)
        if random.random() < modification_prob:
            new_initial_maxpool = random.choice(self.initial_maxpool_choices)

        for i in range(len(new_stage_block_counts)):
            if random.random() < modification_prob:
                new_stage_block_counts[i] = random.choice(self.block_count_choices)

            if random.random() < modification_prob:
                new_stage_block_hps[i].filters = random.choice(self.filter_choices_stage)
            # if random.random() < modification_prob:
            #     new_stage_block_hps[i].stride = random.choice(self.stride_choices)
            if random.random() < modification_prob:
                new_stage_block_hps[i].activation = random.choice(self.activation_choices)

        if random.random() < modification_prob:
            new_classifier_hp.activation = random.choice(self.activation_choices)
        if random.random() < modification_prob:
            new_classifier_hp.dropout_rate = random.choice(self.dropout_choices)
        if random.random() < modification_prob:
            num_mlp_layers = random.choice(self.num_mlp_layers_choices)
            new_classifier_hp.neurons = [random.choice(self.mlp_neuron_choices) for _ in range(num_mlp_layers)]
        elif random.random() < modification_prob and new_classifier_hp.neurons:
             idx_to_modify = random.randrange(len(new_classifier_hp.neurons))
             new_classifier_hp.neurons[idx_to_modify] = random.choice(self.mlp_neuron_choices)

        if len(new_stage_block_counts) != len(new_stage_block_hps):
             print("Warning: Mismatch in stage counts/hps during neighbor generation. Returning original.")
             return reference_hp 

        return ResNetHP(
            initial_conv_hp=new_initial_conv_hp,
            initial_maxpool=new_initial_maxpool,
            stage_block_counts=new_stage_block_counts,
            stage_block_hps=new_stage_block_hps,
            classifier_hp=new_classifier_hp
        )

    # if __name__ == '__main__':
    #     space = ResNetHPSpace()

    #     print("------- Sampling Random ResNet HP -------")
    #     random_hp = space.sample()
    #     print(repr(random_hp))
    #     print("\nRandom HP as Dict:")
    #     print(random_hp.to_dict())

    #     print("\n------- Generating Neighbor of Original ResNet18 HP -------")
    #     neighbor_hp = space.neighbor(original_hp, modification_prob=0.5)
    #     print(repr(neighbor_hp))

    #     print("\n------- Generating Neighbor of Random HP -------")
    #     neighbor_random_hp = space.neighbor(random_hp, modification_prob=0.2)
    #     print(repr(neighbor_random_hp))