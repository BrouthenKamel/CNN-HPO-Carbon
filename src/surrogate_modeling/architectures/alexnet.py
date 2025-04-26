import torch
import torch.nn as nn
import pandas as pd
from utils import get_activation, get_pooling

class CustomAlexNet(nn.Module):
    def __init__(self, csv_path, row, input_channels=3, num_classes=10):
        super().__init__()

        df = pd.read_csv(csv_path)
        config = df.iloc[row].to_dict()

        self.features = self._build_conv_layers(config, input_channels)
        self.classifier = self._build_fc_layers(config, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.classifier(x)
        return x

    def _build_conv_layers(self, config, input_channels):
        layers = []
        in_channels = input_channels
        current_h = current_w = 224  # Assuming input image size for MNIST (28x28)

        for i in range(1, 6):
            out_channels = int(config[f'clb{i}_kernels'])
            ks = int(config[f'clb{i}_kernel_size'])
            stride = int(config[f'clb{i}_stride'])

            padding = ks // 2
            layers.append(nn.Conv2d(in_channels, out_channels, ks, stride=stride, padding=padding))
            layers.append(get_activation(config[f'clb{i}_activation']))

            # Update current feature map size after convolution
            current_h = (current_h - ks + 2 * padding) // stride + 1
            current_w = (current_w - ks + 2 * padding) // stride + 1

            if i in {1, 2}:  # Local Response Normalization for layers 1 and 2
                n = int(config[f'clb{i}_lrn_n'])
                k = float(config[f'clb{i}_lrn_k'])
                alpha = float(config[f'clb{i}_lrn_alpha'])
                beta = float(config[f'clb{i}_lrn_beta'])
                layers.append(nn.LocalResponseNorm(n, alpha=alpha, beta=beta, k=k))

            if i in {1, 2, 5}:  # Only layers 1, 2, and 5 have pooling
                pool_type = config[f'clb{i}_pool_type']
                pool_size = int(config[f'clb{i}_pool_size'])
                pool_stride = int(config[f'clb{i}_pool_stride'])

                # Check if pooling is possible (feature map size >= pool size)
                if current_h >= pool_size and current_w >= pool_size:
                    layers.append(get_pooling(pool_type, pool_size, pool_stride))

                    # Update feature map size after pooling
                    current_h = (current_h - pool_size) // pool_stride + 1
                    current_w = (current_w - pool_size) // pool_stride + 1

            in_channels = out_channels

        return nn.Sequential(*layers)

    def _build_fc_layers(self, config, num_classes):
        fc_layers = []
        in_features = self._get_flattened_dim()

        for i in range(1, 5):
            neurons = int(config[f'fc{i}_neurons'])
            if neurons == 0:
                continue

            fc = nn.Linear(in_features, neurons)

            init = config[f'fc{i}_init'].lower()
            if init == 'xavier':
                nn.init.xavier_uniform_(fc.weight)
            elif init == 'kaiming':
                nn.init.kaiming_uniform_(fc.weight, nonlinearity='relu')

            fc_layers.append(fc)

            dropout = float(config[f'fc{i}_dropout'])
            if dropout > 0:
                fc_layers.append(nn.Dropout(dropout))

            fc_layers.append(nn.ReLU(inplace=True))

            in_features = neurons

        # Always add final classification layer
        fc_layers.append(nn.Linear(in_features, num_classes))
        return nn.Sequential(*fc_layers)

    def _get_flattened_dim(self):
        dummy = torch.zeros(1, 1, 224, 224)  # For MNIST (28x28 input)
        with torch.no_grad():
            out = self.features(dummy)
        return out.view(1, -1).shape[1]