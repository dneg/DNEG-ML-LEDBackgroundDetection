from dataclasses import dataclass

from dneg_ml_toolkit.src.Data.Transforms.BASE_Transform.BASE_Transform_config import BASE_TransformConfig


@dataclass
class ApplyAlphaConfig(BASE_TransformConfig):
    """
    Config dataclass for the Foreground Transform

    """

    AlphaChannel: str = "target"