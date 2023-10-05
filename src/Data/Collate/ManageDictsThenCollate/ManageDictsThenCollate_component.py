from typing import Tuple, List, cast
from dneg_ml_toolkit.src.Data.ml_toolkit_dictionary import MLToolkitDictionary
from dneg_ml_toolkit.src.Data.Transforms.BASE_Transform.BASE_Transform_config import BASE_TransformConfig

#DNEG ML Toolkit and ML Project Imports
from dneg_ml_toolkit.src.Data.Collate.ToolkitStandardCollate import ToolkitStandardCollate
from .ManageDictsThenCollate_config import ManageDictsThenCollateConfig

#External Imports


# Before Collating moves a specified fields to [0]['data'] and [1]['target']
class ManageDictsThenCollate(ToolkitStandardCollate):

    def __init__(self, config: ManageDictsThenCollateConfig, transform_config_list: List[BASE_TransformConfig]):
        super().__init__(config, transform_config_list)
        # Inform the type checker that the config is of type ManageDictsThenCollateConfig
        self.config: ManageDictsThenCollateConfig = cast(ManageDictsThenCollateConfig, config)

    def __call__(self, batch: List[Tuple[MLToolkitDictionary, MLToolkitDictionary]]) \
            -> Tuple[MLToolkitDictionary, MLToolkitDictionary]:
        
        for dicts in batch:
            data = dicts[self.config.InputDict].pop(self.config.InputField)
            target = dicts[self.config.TargetDict].pop(self.config.TargetField)
            dicts[0]['data'] = data
            dicts[1]['target'] = target

        return super().__call__(batch)

        pass
           


