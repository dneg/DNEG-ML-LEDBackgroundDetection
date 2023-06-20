
from typing import List, Optional
from dneg_ml_toolkit.src.Networks.layers import Convolution2D, ConvolutionTranspose2D, get_activation, ActivationType
from dneg_ml_toolkit.src.Networks.BASE_Network.BASE_Network_component import BASE_Network
from dneg_ml_toolkit.src.Data.ml_toolkit_dictionary import MLToolkitDictionary
from src.Networks.UnetV1.UnetV1_config import UnetV1Config

from torch import nn, concat, sigmoid


class UnetV1(BASE_Network):
    """
    Simple implementation based on https://github.com/gntoni/unet_pytorch/blob/master/unet.py

    DNEG discriminator at https://github.com/dneg/DNEG-ML-DeepFakes/blob/develop/src/Networks/UnetV1/UnetV1_component.py
    (Inspired by https://arxiv.org/abs/2002.12655 "A U-Net Based Discriminator for Generative Adversarial Networks")
    """

    def __init__(self, config: UnetV1Config, input_shape: List[int]):
        super().__init__(config, input_shape)

        self.config: UnetV1Config = config
        input_channels = input_shape[2]
        output_channels = config.NumOutputs
        activation = get_activation(ActivationType.ReLU)

        self.conv64 = Convolution2D(input_channels, 64, 3, padding=1, activation=activation)
        self.conv128 = Convolution2D(64, 128, 3, padding=1, activation=activation)
        self.conv256 = Convolution2D(128, 256, 3, padding=1, activation=activation)
        self.conv512 = Convolution2D(256, 512, 3, padding=1, activation=activation)
        self.conv1024 = Convolution2D(512, 1024, 3, padding=1, activation=activation)
        self.upconv1024 = ConvolutionTranspose2D(1024, 512, 2, padding=0, stride=2)
        self.dconv1024 = Convolution2D(1024, 512, 3, padding=1, activation=activation)
        self.upconv512 = ConvolutionTranspose2D(512, 256, 2, padding=0, stride=2)
        self.dconv512 = Convolution2D(512, 256, 3, padding=1, activation=activation)
        self.upconv256 = ConvolutionTranspose2D(256, 128, 2, padding=0, stride=2)
        self.dconv256 = Convolution2D(256, 128, 3, padding=1, activation=activation)
        self.upconv128 = ConvolutionTranspose2D(128, 64, 2, padding=0, stride=2)
        self.dconv128 = Convolution2D(128, 64, 3, padding=1, activation=activation)
        self.conv1 = Convolution2D(64, output_channels, 1, padding=0)
        self.pool = nn.MaxPool2d(2, 2)


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
        x = train_dict['data']

        x1 = self.conv64(x)
        x2 = self.conv128(self.pool(x1))
        x3 = self.conv256(self.pool(x2))
        x4 = self.conv512(self.pool(x3))
        x5 = self.conv1024(self.pool(x4))
        ux5 = self.upconv1024(x5)
        cc5 = concat([ux5, x4], 1)
        dx4 = self.dconv1024(cc5)
        ux4 = self.upconv512(dx4)
        cc4 = concat([ux4, x3], 1)
        dx3 = self.dconv512(cc4)
        ux3 = self.upconv256(dx3)
        cc3 = concat([ux3, x2], 1)
        dx2 = self.dconv256(cc3)
        ux2 = self.upconv128(dx2)
        cc2 = concat([ux2, x1], 1)
        dx1 = self.dconv128(cc2)
        last = self.conv1(dx1)

        train_dict["data"] = sigmoid(last)
        return train_dict
