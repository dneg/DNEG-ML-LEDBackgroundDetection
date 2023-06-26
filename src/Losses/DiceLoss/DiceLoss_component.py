from dneg_ml_toolkit.src.Losses.BASE_Loss.BASE_Loss_component import BASE_Loss
from dneg_ml_toolkit.src.Data.ml_toolkit_dictionary import MLToolkitDictionary
from src.Losses.DiceLoss.DiceLoss_config import DiceLossConfig

from torch import Tensor


class DiceLoss(BASE_Loss):
    """
    Wrapper around Torch's BinaryCrossEntropy Loss that allows it to be configured with parameters
    set from the json experiment configuration.
    """

    def __init__(self, config: DiceLossConfig):
        super().__init__(config)
        self.config: DiceLossConfig = config

    def forward(self, outputs: MLToolkitDictionary, targets: MLToolkitDictionary) -> Tensor:

        assert self.config.Source in outputs, "Cannot use '{0}' as Source for calculating Dice Loss: " \
                                              "'{0}' was not found in the network " \
                                              "outputs dictionary.".format(self.config.Source)
        assert self.config.Target in targets, "Cannot use '{0}' as Target for calculating Dice Loss: " \
                                              "'{0}' was not found in the " \
                                              "target data dictionary.".format(self.config.Target)


        #get flattened versions of data
        model_output = outputs[self.config.Source].view(-1)
        target = targets[self.config.Target].view(-1)


        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)       
        
        intersection = (model_output * target).sum()                            
        dice = (2.*intersection + self.config.Smooth) / (model_output.sum() + target.sum() + self.config.Smooth)  
        
        return 1 - dice
