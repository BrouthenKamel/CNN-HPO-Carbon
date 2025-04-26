import torch
import torch.nn as nn
import torch.nn.functional as F
import random


# Activation mapping
activation_map = {
    "relu": nn.ReLU,
    "leaky relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "tanh": nn.Tanh,
    "linear": nn.Identity
}

# Pooling mapping
pooling_map = {
    "MAX": nn.MaxPool2d,
    "AVG": nn.AvgPool2d
}

class ConfigurableCNN(nn.Module):
    def __init__(self, search_space, config, input_shape=(3, 64, 64), num_classes=10):
        super(ConfigurableCNN, self).__init__()
        self.search_space = search_space
        self.config = config
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.features = self._make_conv_layers()
        self.flatten_dim = self._get_flatten_dim()
        self.classifier = self._make_fc_layers()

    def _make_conv_layers(self):
        layers = []
        in_channels = self.input_shape[0]\
        # for every conv_block do the following : 
        for block_key in sorted(self.config["conv_blocks"].keys()):
            block = self.config["conv_blocks"][block_key]
            # first add as many conv layers as specified by num_convs, all share the same hyperparameters
            for _ in range(block["Conv"]["num_convs"]):
                layers.append(nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=block["Conv"]["kc"],
                    kernel_size=block["Conv"]["ks"],
                    stride=block["Conv"]["s"],
                    padding=block["Conv"]["ks"] // 2 if block["Conv"]["p"] == "SAME" else 0
                ))
                # Add Batch Normalization if enabled in config
                if block.get("BatchNorm", False):  # Check if BatchNorm is enabled
                    layers.append(nn.BatchNorm2d(block["Conv"]["kc"]))
                layers.append(activation_map[block["Conv"]["a"]]())
                in_channels = block["Conv"]["kc"]
            # after adding the conv layers, add a pooling layer
            layers.append(pooling_map[block["Pool"]["pt"]](
                kernel_size=block["Pool"]["ksp"],
                stride=block["Pool"]["sp"]
            ))
            # and finally add a dropout layer
            layers.append(nn.Dropout(block["Pool"]["dp"]))
        return nn.Sequential(*layers)

    def _get_flatten_dim(self):
        with torch.no_grad():
            x = torch.zeros(1, *self.input_shape)
            x = self.features(x)
            return x.view(1, -1).size(1)

    def _make_fc_layers(self):
        layers = []
        in_features = self.flatten_dim
        fc_blocks = self.config["fully_connected_blocks"]
        for key in sorted(fc_blocks.keys()):
            block = fc_blocks[key]
            layers.append(nn.Linear(in_features, block["uf"]))
            layers.append(activation_map[block["af"]]())
            layers.append(nn.Dropout(block["df"]))
            in_features = block["uf"]
        layers.append(nn.Linear(in_features, self.num_classes))  # Final classification layer
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    


    def generate_neighbouring_config(self, to_modify=['Conv', 'fully_connected', 'training']):
        neighbour_config = self.config.copy()

        # Handle convolution blocks
        if 'conv_blocks' in self.config and 'Conv' in to_modify:
            for block_key in sorted(neighbour_config["conv_blocks"].keys()):
                block = neighbour_config["conv_blocks"][block_key]
                p = random.uniform(0, 1)
                if p < 0.5:  # Modify randomly chosen hyperparameter
                    hp_choice = random.choice(["Conv", "BatchNorm", "Pool"])
                    if hp_choice == "Conv":
                        self.modify_conv(block["Conv"])
                    elif hp_choice == "BatchNorm":
                        block["BatchNorm"] = not block.get("BatchNorm", False)  # Toggle BatchNorm
                    elif hp_choice == "Pool":
                        self.modify_pool(block["Pool"])

        # Handle fully connected blocks
        if 'fully_connected_blocks' in self.config and 'fully_connected' in to_modify:
            for block_key in sorted(neighbour_config["fully_connected_blocks"].keys()):
                block = neighbour_config["fully_connected_blocks"][block_key]
                p = random.uniform(0, 1)
                if p < 0.5:  # Modify randomly chosen hyperparameter
                    self.modify_fc(block)

        # Handle training parameters
        if 'training' in self.config and 'training' in to_modify:
            p = random.uniform(0, 1)
            if p < 0.5:  # Modify training hyperparameters
                block = neighbour_config["training"]
                self.modify_training(block)

        return neighbour_config

    def modify_conv(self, conv_params):
        # Modify one of the convolutional hyperparameters
        list_of_params = list(self.search_space["convolutional"].keys())
        param_choice = random.choice(list_of_params)
        self.modify_value(conv_params, param_choice, self.search_space["convolutional"][param_choice])

    def modify_pool(self, pool_params):
        list_of_params = list(self.search_space["pooling"].keys())
        # Modify one of the pooling hyperparameters
        param_choice = random.choice(list_of_params)
        self.modify_value(pool_params, param_choice, self.search_space["pooling"][param_choice])

    def modify_fc(self, fc_params):

        list_of_params = list(self.search_space["fully_connected"].keys())
        # Modify one of the fully connected hyperparameters
        param_choice = random.choice(list_of_params)
        self.modify_value(fc_params, param_choice, self.search_space["fully_connected"][param_choice])

    def modify_training(self, training_params):

        list_of_params = list(self.search_space["training"].keys())
        param_choice = random.choice(list_of_params)
        self.modify_value(training_params, param_choice, self.search_space["training"][param_choice])

    def modify_value(self, params, param_name, value_set):
        if isinstance(value_set, list):
            if params[param_name] in value_set :
                current_value = params[param_name]
                current_index = value_set.index(current_value)
                new_offset = random.choice([-1,1])
                if current_index+new_offset >= 0 and current_index+new_offset < len(value_set) :
                    params[param_name] = value_set[current_index+new_offset]
            else : 
                params[param_name] = random.choice(value_set)


# Define a test search space
search_space = {
    "convolutional": {
        "kc": [32, 64, 96, 100, 128, 160, 192, 224, 256],
        "ks": [3, 5, 7],
        "f": ["relu", "leaky relu", "elu", "tanh", "linear"],
        "s": 1,
        "p": "SAME"
    },
    "batch_normalization": {
        "bn": [True, False]
    },
    "pooling": {
        "dp": [0.3, 0.4, 0.5],
        "ksp": [2, 3],
        "pt": ["MAX", "AVG"],
        "sp": 2
    },
    "fully_connected": {
        "uf": [16, 32, 64, 128, 256, 512],
        "df": [0.1, 0.2, 0.3, 0.4, 0.5],
        "af": ["relu", "leaky relu", "elu", "tanh", "linear"]
    },
    "training": {
        "batch_size": [64, 128, 256],
        "learning_rate": [0.0001, 0.0002, 0.0005, 0.0008, 0.001, 0.002, 0.004, 0.005, 0.008, 0.01],
        "epochs": [10, 20, 30, 50, 100],
        "optimizer": ["adam"],
        "loss_function": ["categorical_crossentropy", "binary_crossentropy", "mse", "mae"]
    }
}

# Example initial configuration
config = {
    "conv_blocks": {
        "Conv_Block_1": {
            "Conv": {
                "num_convs": 1,
                "ks": 5,
                "kc": 32,
                "p": "SAME",
                "s": 1,
                "a": "relu"
            },
            "BatchNorm": False,
            "Pool": {
                "ksp": 3,
                "sp": 2,
                "pt": "MAX",
                "dp": 0.2
            }
        },
        "Conv_Block_2": {
            "Conv": {
                "num_convs": 1,
                "ks": 3,
                "kc": 64,
                "p": "SAME",
                "s": 1,
                "a": "relu"
            },
            "BatchNorm": True,
            "Pool": {
                "ksp": 3,
                "sp": 2,
                "pt": "MAX",
                "dp": 0.4
            }
        }
    },
    "fully_connected_blocks": {
        "Fully_Block_1": {
            "uf": 128,
            "df": 0.5,
            "af": "relu"
        },
        "Fully_Block_2": {
            "uf": 128,
            "df": 0.5,
            "af": "relu"
        },
        "Fully_Block_3": {
            "uf": 128,
            "df": 0.5,
            "af": "relu"
        },
        "Fully_Block_4": {
            "uf": 128,
            "df": 0.5,
            "af": "relu"
        }
    },
    "training": {
        "batch_size": 128,
        "learning_rate": 0.001,
        "epochs": 50,
        "optimizer": "adam",
        "loss_function": "categorical_crossentropy"
    }
}

# Create a ConfigurableCNN instance
model = ConfigurableCNN(search_space, config)

print("Original Configuration:\n", config)

# Generate a neighboring configuration
neighbour_config = model.generate_neighbouring_config(to_modify=['Conv', 'fully_connected', 'training'])

# Print the modified configuration

print("\nNeighbouring Configuration:\n", neighbour_config)