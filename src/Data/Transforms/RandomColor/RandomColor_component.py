from typing import cast, Any, Optional, Dict, Tuple
from dneg_ml_toolkit.src.Data.Transforms.BASE_Transform.BASE_Transform_component import BASE_Transform
from dneg_ml_toolkit.src.Data.ml_toolkit_dictionary import MLToolkitDictionary
from dneg_ml_toolkit.src.Data.image_tools import image_dtype_utils

from src.Data.Transforms.RandomColor.RandomColor_config import RandomColorConfig

import torchvision.transforms as transforms

import numpy as np


class RandomColor(BASE_Transform):

    def __init__(self, config: RandomColorConfig):
        # The Transform will apply the data augmentation to any data specified in its configuration's
        # ApplyTo field.
        # It may specify an additional list of keys and pass it to the parent constructor as the
        # required_transform_metadata parameter. These data items will always be retrieved and are used
        # for calculating the augmentation to apply to the ApplyTo data. These metadata items will not
        # be altered by the Transform
        # This example just requests the dataset index for the piece of data
        required_transform_metadata = ["index"]
        super().__init__(config, required_transform_metadata=required_transform_metadata)


    def apply_transform(self, data_key: str, data_to_transform: Any, transform_metadata: Optional[Dict[str, Any]]) \
            -> Tuple[Any, Optional[MLToolkitDictionary[Any]]]:
        """
        Apply the Transform to the data

        Args:
             data_key: The key that identifies the data
            data_to_transform: The piece of data to apply the transformation to
            transform_metadata: An optional dictionary of additional data gathered using the keys passed as
                required_transform_metadata in the constructor. This additional data does not get
                transformed, but can be used for the calculation and application of the transform
                to the data_to_transform.

        Returns:
            The transformed piece of data; an optional dictionary of additional data that will be added to the same
                dictionary that the transformed data is stored in, for cases where the Transform generates new data from
                the input data.
        """

        # 1. Get the data type of the input data (and the device if it is a tensor). These are tracked so that
        # the transformed data can be returned with the matching data type

        # format using the ML Toolkit's helper function
        data_to_transform, _ = image_dtype_utils.transform_data_type(data_to_transform,
                                                                     to_type=image_dtype_utils.ImageDataType.PILImage)

        # 4. Perform the transformation of the data (Color first)
        transform = transforms.ColorJitter(brightness=self.config.Brightness,
                                           contrast=self.config.Contrast,
                                           saturation=self.config.Saturation,
                                           hue=self.config.Hue)
        transformed_data = transform(data_to_transform)

        # 4.1 Perform the transformation of the data (add monochrome noise)
        if self.config.MonochromeSigma > 0:
            data_to_transform, _ = image_dtype_utils.transform_data_type(transformed_data,
                                                                         to_type=image_dtype_utils.ImageDataType.NPArray)
            monochrome_shape = (data_to_transform.shape[0], data_to_transform.shape[1])
            amount_of_noise = np.random.random() * self.config.MonochromeSigma,
            noise = np.random.normal(0, amount_of_noise, monochrome_shape)
            transformed_data = data_to_transform.astype('float') + noise[:,:,None]
            transformed_data = np.clip(transformed_data, 0, 255).astype('uint8')

        # 4.2 multiply in a tint of a random color
        if self.config.Tint > 0:
            data_to_transform, _ = image_dtype_utils.transform_data_type(transformed_data,
                                                                         to_type=image_dtype_utils.ImageDataType.NPArray)
            data_to_transform = data_to_transform.astype('float')

            amount_of_tint = np.random.random() * self.config.Tint
            randomColor = np.array([1.0, np.random.random(), 0.0])
            randomColor = randomColor + 1 - (np.sum(randomColor)/3)
            np.random.shuffle(randomColor)
            tinted_data = data_to_transform * randomColor
            transformed_data = data_to_transform * (1-amount_of_tint) + tinted_data * amount_of_tint
            transformed_data = np.clip(transformed_data, 0, 255).astype('uint8')


        # 5. Convert the transformed image back to the original Data type.
        #transformed_data, _ = image_dtype_utils.transform_data_type(grayscale_image, to_type=input_datatype,
        #                                                            device=device)
        additional_data = MLToolkitDictionary({})

        return transformed_data, additional_data
