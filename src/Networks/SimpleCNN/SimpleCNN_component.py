from typing import List
import math

from dneg_ml_toolkit.src.Networks.BASE_Network.BASE_Network_component import BASE_Network
import dneg_ml_toolkit.src.Networks.layers as dneg_ml
from dneg_ml_toolkit.src.Data.ml_toolkit_dictionary import MLToolkitDictionary
from src.Networks.SimpleCNN.SimpleCNN_config import SimpleCNNConfig

import torch
import torch.nn as nn


class SimpleCNN(BASE_Network):
    """
    A simple convolutional neural network, built using the ML Toolkit helper functions for creating
    network layers and tracking them by name.

    Args:
        config: The SimpleCNN's configuration object
        input_shape: Data shape in [H,W,C] format
    """

    def __init__(self, config: SimpleCNNConfig, input_shape: List[int]):
        super().__init__(config, input_shape)

        # Inform the type checker that the config is of type ExampleCNNConfig
        self.config: SimpleCNNConfig = config

        assert self.config.NumOutputs is not None, "NumOutputs must have a valid positive value"

        # Always use ReLU activations for the convolutions. See the ExtendedSimpleCNN for an example of how we
        # can make this configurable
        activation = nn.ReLU

        conv_1 = dneg_ml.Convolution2D(in_channels=input_shape[2], out_channels=8, kernel_size=3, stride=1, padding=1,
                                       activation=activation)
        # DNEG ML Toolkit's Networks track each layer by name, accessible through the layers dictionary
        self.add_layer("conv_1", conv_1)
        # Track the output shape of each layer to inform the next layer of its input shape
        output_shape = conv_1.get_output_shape(input_shape=self.input_shape)

        conv_2 = dneg_ml.Convolution2D(in_channels=output_shape[2], out_channels=8, kernel_size=3, stride=1, padding=1,
                                       activation=activation)
        self.add_layer("conv_2", conv_2)
        output_shape = conv_2.get_output_shape(input_shape=output_shape)

        pool_1 = dneg_ml.MaxPooling2D(2)
        self.add_layer("pool_1", pool_1)
        output_shape = pool_1.get_output_shape(input_shape=output_shape)

        conv_3 = dneg_ml.Convolution2D(in_channels=output_shape[2], out_channels=8, kernel_size=3, stride=1, padding=1,
                                       activation=activation)
        self.add_layer("conv_3", conv_3)
        output_shape = conv_3.get_output_shape(input_shape=output_shape)

        pool_2 = dneg_ml.MaxPooling2D(2)
        self.add_layer("pool_2", pool_2)
        output_shape = pool_2.get_output_shape(input_shape=output_shape)

        conv_4 = dneg_ml.Convolution2D(in_channels=output_shape[2], out_channels=16, kernel_size=3, stride=1, padding=1,
                                       activation=activation)
        self.add_layer("conv_4", conv_4)
        output_shape = conv_4.get_output_shape(input_shape=output_shape)

        flatten = dneg_ml.Flatten()
        self.add_layer("flatten", flatten)
        output_shape = flatten.get_output_shape(output_shape)

        self.add_layer("linear", nn.Linear(output_shape[0], self.config.NumOutputs))

        # Since there are no branching paths in this network, we can just add all layers to a Sequential and have
        # a simple forward pass
        self.network = nn.Sequential(*self.layers.values())

        # Call this after self.network is created, as it applies to the submodules of this class
        self.init_layer_weights()

    def forward(self, train_dict: MLToolkitDictionary) -> MLToolkitDictionary:
        """
        Perform the forward pass on the network.

        Args:
            train_dict: All data is transported through ML Toolkit systems in ML Toolkit dictionaries (a custom
                dictionary for holding Tensors). This provides flexibility for training, as multiple tensors can
                be passed into the forward pass at the same time. The ML Toolkit standard is for the Dataset (see
                CIFAR10 or FashionMNIST) to store the core tensor, such as the image in this case, under the "data"
                keyword, and the ground truth under the "target" keyword.

        Returns:
            The input ML Toolkit dictionary, with the "data" field updated with the Network outputs
        """

        x = train_dict["data"]
        train_dict["data"] = self.network(x)  # Replace the network input in-place with the network output
        return train_dict
