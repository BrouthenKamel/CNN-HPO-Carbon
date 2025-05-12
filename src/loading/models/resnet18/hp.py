from typing import List, Optional, Dict, Any

POSSIBLE_ACTIVATIONS = ["ReLU", "Hardswish", "LeakyReLU"]

class ConvLayerHP:
    """Conv + BN + Activation"""
    def __init__(self,
                 filters: int,
                 kernel_size: int,
                 stride: int,
                 activation: str = "ReLU"):
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.activation = activation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "activation": self.activation
        }
    
    def get_flattened_representation(self) -> List[Any]:
        flat = [
            self.filters,
            self.kernel_size,
            self.stride,
        ]
        flat.extend([int(self.activation == act) for act in POSSIBLE_ACTIVATIONS])
        return flat
    
    def __repr__(self):
        return f"ConvLayerHP(filters={self.filters}, kernel={self.kernel_size}, stride={self.stride}, act={self.activation})"

class BasicBlockHP:
    """ResNet Basic Block."""
    def __init__(self,
                 filters: int,
                 stride: int, 
                 activation: str = "ReLU"):
        self.filters = filters
        self.stride = stride
        self.activation = activation
        self.kernel_size = 3 

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filters": self.filters,
            "stride": self.stride,
            "activation": self.activation
        }
    
    def get_flattened_representation(self) -> List[Any]:
        flat = [
            self.filters,
            self.stride,
        ]
        flat.extend([int(self.activation == act) for act in POSSIBLE_ACTIVATIONS])
        return flat
    
    def __repr__(self):
        return f"BasicBlockHP(filters={self.filters}, stride={self.stride}, act={self.activation})"

class ClassifierHP:
    def __init__(self,
                 neurons: List[int],
                 activation: str = "ReLU",
                 dropout_rate: float = 0.0):
        if not (0.0 <= dropout_rate < 1.0):
            raise ValueError(f"Dropout rate must be between 0.0 and 1.0, got {dropout_rate}")

        self.neurons = neurons
        self.activation = activation
        self.dropout_rate = dropout_rate

    def to_dict(self) -> Dict[str, Any]:
        return {
            "neurons": self.neurons,
            "activation": self.activation,
            "dropout_rate": self.dropout_rate
        }

    def get_flattened_representation(self, max_mlp_layers=1) -> List[Any]:

        flat_neurons = self.neurons[:max_mlp_layers]
        flat_neurons.extend([0] * (max_mlp_layers - len(flat_neurons)))

        flat = flat_neurons
        flat.append(self.dropout_rate)
        flat.extend([int(self.activation == act) for act in POSSIBLE_ACTIVATIONS])
        return flat
    
    def __repr__(self):
        return f"ClassifierHP(neurons={self.neurons}, act={self.activation}, dropout={self.dropout_rate})"

class ResNetHP:
    def __init__(self,
                 initial_conv_hp: ConvLayerHP,
                 initial_maxpool: bool, 
                 stage_block_counts: List[int],
                 stage_block_hps: List[BasicBlockHP], 
                 classifier_hp: ClassifierHP):
        self.initial_conv_hp = initial_conv_hp
        self.initial_maxpool = initial_maxpool
        self.stage_block_counts = stage_block_counts
        self.stage_block_hps = stage_block_hps
        self.classifier_hp = classifier_hp

        if len(stage_block_counts) != len(stage_block_hps):
            raise ValueError("Length of stage_block_counts must match length of stage_block_hps")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "initial_conv_hp": self.initial_conv_hp.to_dict(),
            "initial_maxpool": self.initial_maxpool,
            "stage_block_counts": self.stage_block_counts,
            "stage_block_hps": [hp.to_dict() for hp in self.stage_block_hps],
            "classifier_hp": self.classifier_hp.to_dict()
        }

    def get_flattened_representation(self, max_stages=4, max_blocks_per_stage=4, max_mlp_layers=1) -> List[Any]:
        flat_rep = []

        flat_rep.extend(self.initial_conv_hp.get_flattened_representation())
        flat_rep.append(int(self.initial_maxpool))

        num_stages = len(self.stage_block_counts)
        for i in range(max_stages):
            if i < num_stages:
                block_count = min(self.stage_block_counts[i], max_blocks_per_stage)
                flat_rep.append(block_count)
                flat_rep.extend(self.stage_block_hps[i].get_flattened_representation())
            else:
                flat_rep.append(0) 
                dummy_block_hp = BasicBlockHP(filters=0, stride=0, activation=POSSIBLE_ACTIVATIONS[0])
                flat_rep.extend(dummy_block_hp.get_flattened_representation())

        flat_rep.extend(self.classifier_hp.get_flattened_representation(max_mlp_layers=max_mlp_layers))

        return flat_rep
    
    def __repr__(self):
        stages_repr = "\n    ".join([f"Stage {i}: {count} x {hp}" for i, (count, hp) in enumerate(zip(self.stage_block_counts, self.stage_block_hps))])
        return (f"ResNetHP(\n"
                f"  Initial Conv: {self.initial_conv_hp}\n"
                f"  Initial MaxPool: {self.initial_maxpool}\n"
                f"  Stages:\n    {stages_repr}\n"
                f"  Classifier: {self.classifier_hp}\n"
                f")")

original_hp = ResNetHP(
    initial_conv_hp=ConvLayerHP(filters=64, kernel_size=7, stride=2, activation="ReLU"),
    initial_maxpool=True,
    stage_block_counts=[2, 2, 2, 2], 
    stage_block_hps=[
        BasicBlockHP(filters=64, stride=1, activation="ReLU"), 
        BasicBlockHP(filters=128, stride=2, activation="ReLU"),
        BasicBlockHP(filters=256, stride=2, activation="ReLU"),
        BasicBlockHP(filters=512, stride=2, activation="ReLU") 
    ],
    classifier_hp=ClassifierHP(neurons=[], activation="ReLU", dropout_rate=0.0)
)