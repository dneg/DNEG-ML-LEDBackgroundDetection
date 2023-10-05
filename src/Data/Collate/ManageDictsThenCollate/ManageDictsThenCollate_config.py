# Python Built-in Imports
from dataclasses import dataclass, field
from typing import Optional

#DNEG ML Toolkit and ML Project Imports
from dneg_ml_toolkit.src.Component import EMPTY
from dneg_ml_toolkit.src.Data.Collate.ToolkitStandardCollate import ToolkitStandardCollateConfig

#External Imports


@dataclass
class ManageDictsThenCollateConfig(ToolkitStandardCollateConfig):
    """
    TODO Add Docstring, including descriptions of any fields under Attributes.
    Attributes:
        
    """

    InputDict: int = 0
    InputField: str = "data"
    TargetDict: int = 1
    TargetField: str = "target"
