        
# inverted_residual_configs = [
                
#     InvertedResidualConfig(
#         use_se = True,
#         se_config = SqueezeExcitationConfig(squeeze_factor=4, activation_layer="Hardsigmoid"),
#         conv_bn_activation_config = ConvBNActivationConfig(
#             in_channels=16,
#             out_channels=16,
#             kernel_size=3,
#             stride=2,
#             padding=1,
#             norm_layer="BatchNorm2d",
#             activation_layer="ReLU"
#         ),
#         expand_channels=16,
#     ),
                
#     InvertedResidualConfig(
#         use_se = False,
#         se_config = None,
#         conv_bn_activation_config = ConvBNActivationConfig(
#             in_channels=16,
#             out_channels=24,
#             kernel_size=3,
#             stride=2,
#             padding=1,
#             norm_layer="BatchNorm2d",
#             activation_layer="ReLU"
#         ),
#         expand_channels=72,
#     ),
                
#     InvertedResidualConfig(
#         use_se = False,
#         se_config = None,
#         conv_bn_activation_config = ConvBNActivationConfig(
#             in_channels=24,
#             out_channels=24,
#             kernel_size=3,
#             stride=1,
#             padding=1,
#             norm_layer="BatchNorm2d",
#             activation_layer="Hardswish"
#         ),
#         expand_channels=88,
#     ),
                
#     InvertedResidualConfig(
#         use_se = True,
#         se_config = SqueezeExcitationConfig(squeeze_factor=4, activation_layer="Hardsigmoid"),
#         conv_bn_activation_config = ConvBNActivationConfig(
#             in_channels=24,
#             out_channels=40,
#             kernel_size=5,
#             stride=2,
#             padding=2,
#             norm_layer="BatchNorm2d",
#             activation_layer="Hardswish"
#         ),
#         expand_channels=96,
#     ),
                
#     InvertedResidualConfig(
#         use_se = True,
#         se_config = SqueezeExcitationConfig(squeeze_factor=4, activation_layer="Hardsigmoid"),
#         conv_bn_activation_config = ConvBNActivationConfig(
#             in_channels=40,
#             out_channels=40,
#             kernel_size=5,
#             stride=1,
#             padding=2,
#             norm_layer="BatchNorm2d",
#             activation_layer="Hardswish"
#         ),
#         expand_channels=240,
#     ),
    
#     InvertedResidualConfig(
#         use_se = True,
#         se_config = SqueezeExcitationConfig(squeeze_factor=4, activation_layer="Hardsigmoid"),
#         conv_bn_activation_config = ConvBNActivationConfig(
#             in_channels=40,
#             out_channels=40,
#             kernel_size=5,
#             stride=1,
#             padding=2,
#             norm_layer="BatchNorm2d",
#             activation_layer="Hardswish"
#         ),
#         expand_channels=240,
#     ),
    
#     InvertedResidualConfig(
#         use_se = True,
#         se_config = SqueezeExcitationConfig(squeeze_factor=4, activation_layer="Hardsigmoid"),
#         conv_bn_activation_config = ConvBNActivationConfig(
#             in_channels=40,
#             out_channels=48,
#             kernel_size=5,
#             stride=1,
#             padding=2,
#             norm_layer="BatchNorm2d",
#             activation_layer="Hardswish"
#         ),
#         expand_channels=120,
#     ),
    
#     InvertedResidualConfig(
#         use_se = True,
#         se_config = SqueezeExcitationConfig(squeeze_factor=4, activation_layer="Hardsigmoid"),
#         conv_bn_activation_config = ConvBNActivationConfig(
#             in_channels=48,
#             out_channels=48,
#             kernel_size=5,
#             stride=1,
#             padding=2,
#             norm_layer="BatchNorm2d",
#             activation_layer="Hardswish"
#         ),
#         expand_channels=144,
#     ),
    
#     InvertedResidualConfig(
#         use_se = True,
#         se_config = SqueezeExcitationConfig(squeeze_factor=4, activation_layer="Hardsigmoid"),
#         conv_bn_activation_config = ConvBNActivationConfig(
#             in_channels=48,
#             out_channels=96,
#             kernel_size=5,
#             stride=1,
#             padding=2,
#             norm_layer="BatchNorm2d",
#             activation_layer="Hardswish"
#         ),
#         expand_channels=288,
#     ),
    
#     InvertedResidualConfig(
#         use_se = True,
#         se_config = SqueezeExcitationConfig(squeeze_factor=4, activation_layer="Hardsigmoid"),
#         conv_bn_activation_config = ConvBNActivationConfig(
#             in_channels=96,
#             out_channels=96,
#             kernel_size=5,
#             stride=1,
#             padding=2,
#             norm_layer="BatchNorm2d",
#             activation_layer="Hardswish"
#         ),
#         expand_channels=576,
#     ),
                
#     InvertedResidualConfig(
#         use_se = True,
#         se_config = SqueezeExcitationConfig(squeeze_factor=4, activation_layer="Hardsigmoid"),
#         conv_bn_activation_config = ConvBNActivationConfig(
#             in_channels=96,
#             out_channels=96,
#             kernel_size=5,
#             stride=1,
#             padding=2,
#             norm_layer="BatchNorm2d",
#             activation_layer="Hardswish"
#         ),
#         expand_channels=576,
#     ),
# ]

# config = MobileNetConfig(
#     initial_conv_config=ConvBNActivationConfig(
#         in_channels=3,
#         out_channels=16,
#         kernel_size=3,
#         stride=2,
#         padding=1,
#         activation_layer='Hardswish',
#         norm_layer='BatchNorm2d',
#         eps=1e-3,
#         momentum=1e-2,
#         ignore_in_channels=True
#     ),
#     inverted_residual_configs=inverted_residual_configs,
#     last_conv_upsample=6,
#     last_conv_config=ConvBNActivationConfig(
#         in_channels=inverted_residual_configs[-1].conv_bn_activation_config.out_channels,
#         out_channels=inverted_residual_configs[-1].conv_bn_activation_config.out_channels * 6,
#         kernel_size=1,
#         stride=1,
#         padding=0,
#         activation_layer='Hardswish',
#         norm_layer='BatchNorm2d',
#         eps=1e-3,
#         momentum=1e-2
#     ),
#     classifier_config=ClassifierConfig(
#         neurons=1024,
#         activation_layer='Hardswish',
#         dropout_rate=0.2
#     )
# )

