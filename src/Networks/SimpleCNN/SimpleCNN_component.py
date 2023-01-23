from typing import List
import math

from dneg_ml_toolkit.src.Networks.BASE_Network.BASE_Network_component import BASE_Network
from dneg_ml_toolkit.src.Networks.layers import Convolution2D, MaxPooling2D, get_activation, ActivationType
from dneg_ml_toolkit.src.Data.ml_toolkit_dictionary import MLToolkitDictionary
from src.Networks.SimpleCNN.SimpleCNN_config import SimpleCNNConfig

import torch
import torch.nn as nn


class SimpleCNN(BASE_Network):
    """

    Args:
        config:
        input_shape: Data shape in [H,W,C] format
    """

    def __init__(self, config: SimpleCNNConfig, input_shape: List[int]):
        super().__init__(config, input_shape)

        # Inform the type checker that the config is of type ExampleCNNConfig
        self.config: SimpleCNNConfig = config

        activation = nn.ReLU

        input_height, input_width, input_channels = self.input_shape

        conv_1 = Convolution2D(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1, activation=activation)
        self.add_layer("conv_1", conv_1)
        output_shape = conv_1.get_output_shape(input_shape=self.input_shape)

        conv_2 = Convolution2D(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1, activation=activation)
        self.add_layer("conv_2", conv_2)
        output_shape = conv_2.get_output_shape(input_shape=output_shape)

        pool_1 = MaxPooling2D(2)
        self.add_layer("pool_1", pool_1)
        output_shape = pool_1.get_output_shape(input_shape=output_shape)

        conv_3 = Convolution2D(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1, activation=activation)
        self.add_layer("conv_3", conv_3)
        output_shape = conv_3.get_output_shape(input_shape=output_shape)

        pool_2 = MaxPooling2D(2)
        self.add_layer("pool_2", pool_2)
        output_shape = pool_2.get_output_shape(input_shape=output_shape)

        conv_4 = Convolution2D(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1,
                               activation=activation)
        self.add_layer("conv_4", conv_4)
        output_shape = conv_4.get_output_shape(input_shape=output_shape)
        output_shape = math.prod(output_shape)

        self.add_layer("flatten", nn.Flatten())
        self.add_layer("linear", nn.Linear(output_shape, self.config.NumOutputs))

        # Since there are no branching paths in this network, can just add all layers to a Sequential and have
        # a simple forward pass
        self.network = nn.Sequential(*self.layers.values())

        # Call this after self.network is created, as it applies to the submodules of this class
        self.init_layer_weights()

    def forward(self, train_dict: MLToolkitDictionary) -> MLToolkitDictionary:
        x = train_dict["data"]
        x = self.network(x)
        train_dict["data"] = x  # Replace the network input in-place with the network output
        return train_dict
