from dataclasses import dataclass
from typing import Optional

from dneg_ml_toolkit.src.Losses.BASE_Loss.BASE_Loss_config import BASE_LossConfig


@dataclass
class DiceLossConfig(BASE_LossConfig):
    """
    Config dataclass for all configuration necessary for BinaryCrossEntropy Loss

    """
    Smooth: Optional[float] = 1
