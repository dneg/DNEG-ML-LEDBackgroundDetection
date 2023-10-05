from typing import Optional, Dict,Any
from dataclasses import dataclass, field
from dneg_ml_toolkit.src.Networks.BASE_Network.BASE_Network_config import BASE_NetworkConfig
from dneg_ml_toolkit.src.Component.component_config import EMPTY
from dneg_ml_toolkit.src.Networks.layers import ActivationType

@dataclass
class UnetV1Config(BASE_NetworkConfig):
    """
    Config dataclass for all configuration necessary for the UnetV1 network.
    """

    # Add any parameters unique to the UNETV1 network here
    NumLayers: int = 5
    BaseChannels: int = 16

    NumOutputs: int = 1

    Activation: ActivationType = ActivationType.LeakyReLU
    ActivationNegativeSlope: Optional[float] = 0.2  # Only needed for LeakyReLU

    # JSON validation rule to ensure that ActivationNegativeSlope is configured when using
    # LeakyReLU activation.
    # For any field, you can create a field with an "_" prefix and "_conditional" postfix to define
    # custom jsonschema rules
    _ActivationNegativeSlope_conditional: Dict[str, Any] = field(
        default_factory=lambda: {
            "if": {
                "properties": {"Activation": {"const": "LeakyReLU"}}
            },
            "then": {
                "required": ["ActivationNegativeSlope"],
                "properties": {"error_message": "ActivationNegativeSlope is required when using LeakyReLU Activation."}
            }

        })
