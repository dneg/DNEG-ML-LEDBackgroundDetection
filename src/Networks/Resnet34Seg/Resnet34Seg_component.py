from typing import List, Optional, Union, cast

from dneg_ml_toolkit.src.Data.ml_toolkit_dictionary import MLToolkitDictionary
import dneg_ml_toolkit.src.Networks.layers as dneg_ml
from dneg_ml_toolkit.src.Networks.BASE_Network import BASE_Network
from .Resnet34Seg_config import Resnet34SegConfig, ResidualType

from dneg_ml_toolkit.src.Networks.layers import Convolution2D, MaxPooling2D, AdaptiveAveragePooling2d, Flatten, _build_stage, ConvolutionTranspose2D

import torch.nn as nn
import torch

class Resnet34Seg(BASE_Network):

    def __init__(self, config: Resnet34SegConfig, input_shape: List[int]):
        super().__init__(config, input_shape)

        self.config: Resnet34SegConfig = cast(Resnet34SegConfig, self.config)

        if self.config.Residual == ResidualType.ClassicResidual:
            residual_block = dneg_ml.ClassicResidualBlock
        elif self.config.Residual == ResidualType.PreActResidual:
            residual_block = dneg_ml.PreActResidualBlock
        else:
            raise ValueError("{} is not a valid ResidualType".format(self.config.Residual))

        activation = dneg_ml.get_activation(self.config.Activation)
        activation_params = {}
        if self.config.Activation == dneg_ml.ActivationType.LeakyReLU:
            activation_params["negative_slope"] = self.config.ActivationNegativeSlope

        stages = [3, 4, 6, 3]
        planes = [64, 64, 128, 256, 512]

        #-------------------------Replaced build_resnet with modified version to add decoder------------------------
        #self.output_shape = dneg_ml.build_resnet(input_shape=input_shape, planes=planes, stages=stages,
                                                 #add_layer_fn=self.add_layer, num_outputs=self.config.NumOutputs,
                                                 #residual_block=residual_block, batch_norm=True,
                                                 #activation=activation, **activation_params)

        #-------------------------Replaced build_resnet with modified version to add decoder------------------------
        batch_norm=True
        conv1 = Convolution2D(in_channels=input_shape[2], out_channels=planes[0], stride=2, kernel_size=7,
                              padding=3,
                              batch_norm=batch_norm, activation=activation, **activation_params)
        self.add_layer("conv_1", conv1)
        output_shape = conv1.get_output_shape(input_shape)

        max_pool1 = MaxPooling2D(3, stride=2, padding=1)
        self.add_layer("max_pool_1", max_pool1)
        output_shape = max_pool1.get_output_shape(output_shape)

        for i in range(len(stages)):
            output_shape = _build_stage(stage_number=i + 1, input_shape=output_shape, stage_size=stages[i],
                                        stage_planes=planes[i + 1], add_layer_fn=self.add_layer,
                                        batch_norm=batch_norm, residual_block=residual_block,
                                        activation=activation, **activation_params)

        #----NEW PART---- we have h/4, w/4, 512.  upconv twice and then conv2d to get 1 channel
        upconv1 = ConvolutionTranspose2D(output_shape[2], output_shape[2], 2, padding=0, stride=2)
        self.add_layer("upconv_1",upconv1)
        output_shape = upconv1.get_output_shape(output_shape)

        upconv2 = ConvolutionTranspose2D(output_shape[2], output_shape[2], 2, padding=0, stride=2)
        self.add_layer("upconv_2",upconv2)
        output_shape = upconv2.get_output_shape(output_shape)

        conv2 = Convolution2D(output_shape[2], self.config.NumOutputs, 1, padding=0)
        self.add_layer("conv_2",conv2)
        output_shape = conv2.get_output_shape(output_shape)

        #----OLD PART----
        #avg_pool = AdaptiveAveragePooling2d((1, 1))
        #self.add_layer("avg_pool_1", avg_pool)
        #output_shape = avg_pool.get_output_shape(output_shape)
    
        #flatten = Flatten()
        #self.add_layer("flatten", flatten)
        #output_shape = flatten.get_output_shape(output_shape)

        #self.add_layer("linear", nn.Linear(output_shape[0], self.config.NumOutputs))
        #output_shape = self.config.NumOutputs
        #-------------------------Replace build_resnet with modified version to add decoder------------------------


        # Since there are no branching paths in this network, can just add all layers to a Sequential and have
        # a simple forward pass
        self.network = nn.Sequential(*self.layers.values())

        # Call this after self.network is created, as it applies to the submodules of this class
        self.init_layer_weights()

    def forward(self, x, step: Optional[int] = -1) \
            -> Union[MLToolkitDictionary, torch.Tensor]:
        """
        Forward pass for the network

        Args:
            x: During training, this is the Training data dictionary, which includes the data and any additional
                training information. In order to support exporting the network (to ONNX, TorchScript etc.), this must
                also accept a tensor for inference.
                Note that strongly typing this parameter to accept either
                tensors or MLToolkitDictionary prevents ONNX export from working correctly.
                Its type should be: Union[MLToolkitDictionary, torch.Tensor]
            step: Allow the trainer to inform the Network of the current step

        Returns:
            During training, returns the input dictionary updated with network outputs. In order to support exporting
            the network, this must also be able to return a tensor if the input is a tensor.
        """

        # When exported, the Network must be able to execute on a Tensor instead of the ML Toolkit's MLToolkitDictionary
        # ONNX export does not support isinstance, but by not strongly
        # typing x:Union[MLToolkitDictionary, torch.Tensor], the exporter appears to automatically infer x's type as
        # torch.Tensor
        if isinstance(x, torch.Tensor):
            batch = x
        else:
            batch = x["data"]
            # This assert is required by torch.jit.script to ensure that each branch of the conditional returns a
            # tensor, even though this branch will never actually execute on the exported Network.
            assert isinstance(batch, torch.Tensor)

        output = self.network(batch)

        if isinstance(x, torch.Tensor):
            x = torch.sigmoid(output)
        else:
            # Update the "data" entry with the network output, preserving any other metadata in the dictionary
            x["data"] = torch.sigmoid(output)  # Replace the network input in-place with the network output

        return x
