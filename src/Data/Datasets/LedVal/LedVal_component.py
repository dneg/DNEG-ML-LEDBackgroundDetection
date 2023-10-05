from typing import List, Optional, cast, Tuple

from dneg_ml_toolkit.src.Data.Datasets.BASE_Dataset.BASE_Dataset_component import BASE_Dataset
from dneg_ml_toolkit.src.Data.ml_toolkit_dictionary import MLToolkitDictionary
from dneg_ml_toolkit.src.Data.Collate.BASE_Collate.BASE_Collate_component import BASE_Collate
from dneg_ml_toolkit.src.Data.Transforms.ToTensor.ToTensor_config import ToTensorConfig
from dneg_ml_toolkit.src.Data.Transforms.ToTensor.ToTensor_component import ToTensor

from src.Data.Datasets.LedVal.LedVal_config import LedValConfig

from PIL import Image, ImageOps


class LedVal(BASE_Dataset):
    """

    Args:
        config: Dataset config object
        collate_component: Reference to the Collate Component. This can be used to access the Collate's Transform
            pipeline, and apply Transforms in the Dataset's getitem if the Collate is configured to not apply
            the Transforms (Transforming in the Dataset is more efficient as it uses the Dataloader's workers,
            but the Transforms cannot be applied with an awareness of other batch elements.)
    """

    def __init__(self, config: LedValConfig, collate_component: BASE_Collate):
        # LedVal returns PIL images, so append a ToTensor Transform to the end of the
        # configured Transforms so that the data is always returned in tensor format.
        collate_component.append_transform(ToTensor(ToTensorConfig(Type="ToTensor", ApplyTo=["data"])))

        super().__init__(config, allow_multiple_sources=False, check_source_exists=False,
                         collate_component=collate_component)

        self.config: LedValConfig = cast(LedValConfig, self.config)

    def __len__(self) -> int:
        return 1048

    def get_item_data(self, index: int) -> Tuple[MLToolkitDictionary, Optional[MLToolkitDictionary]]:
        filename = self.config.Source+'/img/img{:05d}.png'.format(index)
        image = Image.open(filename)
        train_dict = MLToolkitDictionary({"data": image, "index": index})

        # Pass all the data for the sample through the Transform pipeline
        return train_dict, self.get_item_metadata(index)

    def get_item_metadata(self, index: int) -> MLToolkitDictionary:
        targetname = self.config.Source+'/gt/gt{:05d}.png'.format(index)
        target = Image.open(targetname)
        grayTarget = ImageOps.grayscale(target)

        return  MLToolkitDictionary({"target": grayTarget, "index": index, "dataset": "LedVal"})


    def get_data_shape(self) -> List[int]:
        return(128,128,3)