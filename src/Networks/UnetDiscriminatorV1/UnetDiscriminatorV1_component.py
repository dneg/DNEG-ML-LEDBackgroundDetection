
from typing import List, Optional
from dneg_ml_toolkit.src.Networks.layers import Convolution2D, ConvolutionTranspose2D, get_activation, ActivationType
from dneg_ml_toolkit.src.Networks.BASE_Network.BASE_Network_component import BASE_Network
from dneg_ml_toolkit.src.Data.ml_toolkit_dictionary import MLToolkitDictionary
from src.Networks.UnetDiscriminatorV1.UnetDiscriminatorV1_config import UnetDiscriminatorV1Config

from torch import nn, float32, concat, Tensor
from torch.nn import LeakyReLU


class UnetDiscriminatorV1(BASE_Network):
    """
    Simple implementation based on https://github.com/gntoni/unet_pytorch/blob/master/unet.py

    DNEG discriminator at https://github.com/dneg/DNEG-ML-DeepFakes/blob/develop/src/Networks/UnetDiscriminatorV1/UnetDiscriminatorV1_component.py
    (Inspired by https://arxiv.org/abs/2002.12655 "A U-Net Based Discriminator for Generative Adversarial Networks")
    """

    def __init__(self, config: UnetDiscriminatorV1Config, input_shape: List[int]):
        super().__init__(config, input_shape)

        # Inform the type checker that self.config is a UnetDiscriminatorV1Config
        self.config: UnetDiscriminatorV1Config = config
        input_channels = input_shape[2]

        # Configure activation function
        activation = get_activation(self.config.Activation)
        activation_params = {}

        if self.config.Activation == ActivationType.LeakyReLU:
            activation_params["negative_slope"] = self.config.ActivationNegativeSlope

        # 1. Generate variables for layer configuration
        layers = [[3, 2] for _ in range(self.config.NumLayers)]

        level_channels = {
            i - 1: v
            for i, v in enumerate(
                [min(self.config.BaseChannels * (2 ** i), 512) for i in range(len(layers) + 1)]
            )
        }

        # 2.1. Input Conv: Convolution with activation
        in_conv = Convolution2D(in_channels=input_channels,
                                out_channels=level_channels[-1],
                                kernel_size=1,
                                padding="valid",
                                dtype=float32,
                                activation=activation,
                                **activation_params
                                )
        self.add_layer("in_conv", in_conv)
        output_shape = in_conv.get_output_shape(input_shape=input_shape)

        # 2.2. Conv stack and Upconv stack
        convs_layer_names = []  # Track the layer names so the stack can be built into a ModuleList for use in forward()
        upconv_layer_names = []

        conv_out_shape = output_shape
        for i, (kernel_size, strides) in enumerate(layers):
            # 2.2.1. Convolutions with activation
            conv_name = "conv_{}".format(i + 1)
            convs_layer_names.append(conv_name)
            conv = Convolution2D(in_channels=conv_out_shape[2], out_channels=level_channels[i],
                                 kernel_size=kernel_size, stride=strides,
                                 padding="same" if strides == 1 else (kernel_size // strides),
                                 dtype=float32, activation=activation,
                                 **activation_params)
            self.add_layer(conv_name, conv)
            conv_out_shape = conv.get_output_shape(input_shape=conv_out_shape)

            # 2.2.2. Upconv with activation - these are created in reverse order to the convolutions (i.e. first conv
            # matches the last upconv).
            upconv_name = "upconv_{}".format(len(layers) - i)
            upconv_layer_names.insert(0, upconv_name)

            upconv = ConvolutionTranspose2D(in_channels=level_channels[i] * (2 if i != len(layers) - 1 else 1),
                                            out_channels=level_channels[i - 1], kernel_size=kernel_size, stride=strides,
                                            padding=kernel_size // strides, dtype=float32,
                                            activation=activation, **activation_params
                                            )
            self.add_layer(upconv_name, upconv)

        # 2.3. Centre Out: Convolution without activation
        self.add_layer("centre_out", (Convolution2D(in_channels=conv_out_shape[2],
                                                    out_channels=1,
                                                    kernel_size=1,
                                                    padding="valid",
                                                    dtype=float32)))

        # 2.4. Centre Conv: Convolution with activation
        self.add_layer("centre_conv", (Convolution2D(in_channels=conv_out_shape[2],
                                                     out_channels=conv_out_shape[2],
                                                     kernel_size=1,
                                                     padding="valid",
                                                     dtype=float32)))

        # 2.5. Out Conv: Convolution without activation
        self.add_layer("out_conv", (Convolution2D(in_channels=level_channels[-1] * 2,
                                                  out_channels=1,
                                                  kernel_size=1,
                                                  padding="valid",
                                                  dtype=float32)))

        # 3. Split the layers into different variables to use in the forward, since this network is not
        # a standard feed forward network, so they can't be combined into a single Sequential
        self.in_conv = self.layers["in_conv"]
        self.convs = nn.ModuleList([self.layers[key] for key in convs_layer_names])
        self.upconvs = nn.ModuleList([self.layers[key] for key in upconv_layer_names])
        self.centre_out = self.layers["centre_out"]
        self.centre_conv = self.layers["centre_conv"]
        self.out_conv = self.layers["out_conv"]

        # Call this after self.network is created, as it applies to the submodules of this class
        self.init_layer_weights()

    def forward(self, train_dict: MLToolkitDictionary, step: Optional[int] = -1) -> MLToolkitDictionary:
        """
        Perform a forward pass in the discriminator

        Args:
            train_dict: Discriminator input Data

        Returns:
            Dictionary of the discriminator outputs
        """
        x = train_dict["data"]

        x = self.in_conv(x)
        
        encs = []
        for conv in self.convs:
            encs.insert(0, x)
            x = conv(x)  # Convolution with activation

        centre_out, x = self.centre_out(x), self.centre_conv(x)

        for i, (upconv, enc) in enumerate(zip(self.upconvs, encs)):
            x = upconv(x, output_size=enc.size())  # Conv Transpose with activation
            x = concat([enc, x], dim=1)  # Data is in NCHW format, so dim 1 is channels

        x = self.out_conv(x)

        # Combine the outputs into a dictionary
        train_dict["data"] = x
        return train_dict