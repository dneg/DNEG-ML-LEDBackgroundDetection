from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from dneg_ml_toolkit.src.Component.component_config import EMPTY
from dneg_ml_toolkit.src.Networks.BASE_Network.BASE_Network_config import BASE_NetworkConfig
from dneg_ml_toolkit.src.Networks.layers import ActivationType


@dataclass
class SimpleCNNConfig(BASE_NetworkConfig):
    NumOutputs: int = 10  # TODO Set from data
