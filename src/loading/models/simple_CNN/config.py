from src.schema.model import ModelArchitecture
from src.schema.block import CNNBlock, MLPBlock
from src.schema.layer import (
    ConvLayer,
    PoolingLayer,
    DropoutLayer,
    LinearLayer,
    ActivationLayer,
    BatchNormLayer,
    AdaptivePoolingLayer,
    PoolingType,
    ActivationType
)
from src.schema.training import TrainingParams, OptimizerType
from src.loading.models.simple_CNN.hp import SimpleCNNHP

class SimpleCNNConfig:
    """
    Build a ModelArchitecture directly from SimpleCNNHP.
    """
    @staticmethod
    def from_hp(hp: SimpleCNNHP) -> ModelArchitecture:
        # Initial convolution
        initial_conv = None
        if hp.initial_conv:
            ic = hp.initial_conv
            initial_conv = ConvLayer(
                filters=ic.filters,
                kernel_size=ic.kernel_size,
                stride=ic.stride,
                padding=ic.padding
            )

        # CNN blocks
        cnn_blocks = []
        for b in hp.cnn_block_hps:
            conv = ConvLayer(
                filters=b.conv.filters,
                kernel_size=b.conv.kernel_size,
                stride=b.conv.stride,
                padding=b.conv.padding
            )
            activation = (
                ActivationLayer(type=ActivationType[b.activation.upper()])
                if b.activation else None
            )
            pooling = (
                PoolingLayer(
                    type=PoolingType[b.pooling['type'].upper()],
                    kernel_size=b.pooling['kernel_size'],
                    stride=b.pooling['stride'],
                    padding=b.pooling['padding']
                ) if b.pooling else None
            )
            batch_norm = BatchNormLayer() if b.batch_norm else None
            cnn_blocks.append(
                CNNBlock(
                    conv_layer=conv,
                    activation_layer=activation,
                    pooling_layer=pooling,
                    batch_norm_layer=batch_norm
                )
            )

        # Adaptive pooling
        adaptive_pool = None
        if hp.adaptive_pooling:
            adaptive_pool = AdaptivePoolingLayer(
                type=PoolingType.AVG,
                output_size=hp.adaptive_pooling
            )

        # MLP blocks
        mlp_blocks = []
        for m in hp.mlp_block_hps:
            linear = LinearLayer(neurons=m.neurons)
            activation = (
                ActivationLayer(type=ActivationType[m.activation.upper()])
                if m.activation else None
            )
            dropout = DropoutLayer(rate=m.dropout) if m.dropout else None
            mlp_blocks.append(
                MLPBlock(
                    dropout_layer=dropout,
                    linear_layer=linear,
                    activation_layer=activation
                )
            )

        # Training parameters
        tr = hp.training
        training_params = TrainingParams(
            epochs=tr['epochs'],
            batch_size=tr['batch_size'],
            learning_rate=tr['learning_rate'],
            optimizer=OptimizerType[tr['optimizer'].upper()].value,
            momentum=tr.get('momentum'),
            weight_decay=tr.get('weight_decay')
        )

        return ModelArchitecture(
            initial_conv_layer=initial_conv,
            cnn_blocks=cnn_blocks,
            adaptive_pooling_layer=adaptive_pool,
            mlp_blocks=mlp_blocks,
            training_params=training_params
        )
