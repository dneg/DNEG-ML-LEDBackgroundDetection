from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union

from dneg_ml_toolkit.src.Data.Transforms.BASE_Transform.BASE_Transform_config import BASE_TransformConfig


@dataclass
class CombineConfig(BASE_TransformConfig):
    """
    Config dataclass for the Foreground Transform
    """

    Foreground: str = "data"
