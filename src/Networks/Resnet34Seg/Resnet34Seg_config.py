from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from dneg_ml_toolkit.src.Networks.BASE_Network import BASE_NetworkConfig
from dneg_ml_toolkit.src.Networks.layers import ActivationType, ResidualType


@dataclass
class Resnet34SegConfig(BASE_NetworkConfig):
    """

    """

    NumOutputs: int = 1

    Residual: ResidualType = ResidualType.ClassicResidual

    Activation: ActivationType = ActivationType.ReLU
    ActivationNegativeSlope: Optional[float] = None  # Only needed for LeakyReLU

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
