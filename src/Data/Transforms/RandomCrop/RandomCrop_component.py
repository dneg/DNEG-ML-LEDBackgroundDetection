from typing import cast, Any, Optional, Dict, Tuple
from dneg_ml_toolkit.src.Data.Transforms.BASE_Transform.BASE_Transform_component import BASE_Transform
from dneg_ml_toolkit.src.Data.ml_toolkit_dictionary import MLToolkitDictionary
from dneg_ml_toolkit.src.Data.image_tools import image_dtype_utils

from src.Data.Transforms.RandomCrop.RandomCrop_config import RandomCropConfig

import numpy as np
import math


OFFSET = np.array([[0,0], [math.pi,0], [math.pi/2, math.pi/2]]) #locations of blue, green, red peaks 

class RandomCrop(BASE_Transform):

    def __init__(self, config: RandomCropConfig):
        # The Transform will apply the data augmentation to any data specified in its configuration's
        # ApplyTo field.
        # It may specify an additional list of keys and pass it to the parent constructor as the
        # required_transform_metadata parameter. These data items will always be retrieved and are used
        # for calculating the augmentation to apply to the ApplyTo data. These metadata items will not
        # be altered by the Transform
        # This example just requests the dataset index for the piece of data
        required_transform_metadata = []
        super().__init__(config, required_transform_metadata=required_transform_metadata)

        self.crop_size = (config.Size, config.Size)


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
        input_datatype, device = image_dtype_utils.get_image_datatype(data_to_transform)

        # format using the ML Toolkit's helper function
        data_to_transform, _ = image_dtype_utils.transform_data_type(data_to_transform,
                                                                     to_type=image_dtype_utils.ImageDataType.NPArray)
        (w,h,c) = np.shape(data_to_transform) 


        # 4. Perform the transformation of the data

        # Use self.random_seed to seed the rng, so that all calls to apply_transform in the current step
        # will have the same sequence of randoms
        rng = np.random.default_rng(seed=self.random_seed)
        x = math.floor(rng.uniform(0, w-self.config.Size))
        y = math.floor(rng.uniform(0, h-self.config.Size))

        cropped_image = data_to_transform[x:x+self.config.Size, y:y+self.config.Size]

        # 5. Convert the transformed image back to the original Data type.
        transformed_data, _ = image_dtype_utils.transform_data_type(cropped_image, to_type=input_datatype,
                                                                    device=device)
        additional_data = MLToolkitDictionary({})

        return transformed_data, additional_data