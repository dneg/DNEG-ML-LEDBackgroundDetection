from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union

from dneg_ml_toolkit.src.Data.Transforms.BASE_Transform.BASE_Transform_config import BASE_TransformConfig


@dataclass
class RandomColorConfig(BASE_TransformConfig):
    """
    Config dataclass for the Random Color Transform

    """

    # values should range from 0..1, 0 means no change, 1 means a lot of random change (which could be +ve or -ve)
    Brightness: Optional[float] = 0
    Contrast: Optional[float] = 0
    Saturation: Optional[float] = 0

    # hue should range from 0..0.5  
    Hue: Optional[float] = 0

    # amount (in intensity) of monochrome noise to add (sigma is uniformly chosen from 0..given sigma))
    MonochromeSigma: Optional[float] = 0.0  
