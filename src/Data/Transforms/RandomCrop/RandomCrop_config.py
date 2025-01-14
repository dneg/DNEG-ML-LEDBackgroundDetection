from dataclasses import dataclass

from dneg_ml_toolkit.src.Data.Transforms.BASE_Transform.BASE_Transform_config import BASE_TransformConfig
from typing import Optional


@dataclass
class RandomCropConfig(BASE_TransformConfig):
    """
    Config dataclass for the RandomCrop Transform

    """

    Size: int = 128 
    FixedSeed: Optional[int] = None
