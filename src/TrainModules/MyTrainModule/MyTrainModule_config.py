from dataclasses import dataclass, field

from dneg_ml_toolkit.src.Networks.BASE_Network.BASE_Network_config import BASE_NetworkConfig
from dneg_ml_toolkit.src.Losses.BASE_Loss.BASE_Loss_config import BASE_LossConfig
from dneg_ml_toolkit.src.Optimizers.BASE_Optimizer.BASE_Optimizer_config import BASE_OptimizerConfig
from dneg_ml_toolkit.src.Schedulers.BASE_Scheduler.BASE_Scheduler_config import BASE_SchedulerConfig
from dneg_ml_toolkit.src.TrainModules.BASE_TrainModule.BASE_TrainModule_config import BASE_TrainModuleConfig
from dneg_ml_toolkit.src.Data.Transforms.BASE_Transform.BASE_Transform_config import BASE_TransformConfig
from dneg_ml_toolkit.src.Component.component_config import EMPTY

from typing import List, Optional, Union, Dict, Any


@dataclass
class MyTrainModuleConfig(BASE_TrainModuleConfig):
    """
    Config dataclass for all configuration necessary for the My Train Module.

    The My Train Module performs training on a single Network, calculates loss on the Network output
    using 1 or more loss functions, and optimizes with the specified Optimizer.

    Attributes:
        Network (BASE_NetworkConfig): Configuration for a single Network
        Loss (BASE_LossConfig or List[BASE_LossConfig]): Configuration for 1 or more losses.
            Total loss will be aggregated across all losses.
        Optimizer (BASE_OptimizerConfig): Configuration for a single Optimizer.
        Scheduler (Optional BASE_SchedulerConfig): Configuration for an optional LR Scheduler

    """
    # 1. Network configuration
    Network: BASE_NetworkConfig = EMPTY
    # All required Components should be set with EMPTY as the default value. The JSON configuration system will ensure
    # that these are correctly configured, and give guidance if they are not. (i.e. If the Network is not correctly
    # set, the configuration will list all the registered Networks available for use.)

    # 2. Loss configuration - the My Train Module will calculate loss using 1 or more loss function
    # and aggregate the total loss.
    Loss: Union[Optional[List[BASE_LossConfig]], BASE_LossConfig] = EMPTY

    # 3. Optimizer configuration
    Optimizer: BASE_OptimizerConfig = EMPTY
    # 3.1 Optional LR Scheduler for the Optimizer - since it is Optional, this has a default value of None
    Scheduler: Optional[BASE_SchedulerConfig] = None

    # The JSON validation system treats any field that starts with "_" as a private field, so it isn't exposed to
    # the JSON configuration. This allows the Config objects to transport values set elsewhere in the system, after
    # the JSON has been loaded.
    _InputShape: Optional[List[int]] = None

    # Flag the above private field as saveable so the generated value that is stored in it is saved along
    # with the checkpoint and can be loaded for inference after training
    __InputShape_additional_attributes: Dict[str, Any] = field(default_factory=lambda: {"save": True})

    # Property to access the values of the above private field
    @property
    def InputShape(self):
        return self._InputShape
