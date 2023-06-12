from typing import cast, Any, Optional, Dict, Tuple
from dneg_ml_toolkit.src.Data.Transforms.BASE_Transform.BASE_Transform_component import BASE_Transform
from dneg_ml_toolkit.src.Data.ml_toolkit_dictionary import MLToolkitDictionary
from dneg_ml_toolkit.src.Data.image_tools import image_dtype_utils

from src.Data.Transforms.Halftone.Halftone_config import HalftoneConfig

from torchvision.transforms import RandomPerspective, CenterCrop

import numpy as np
import math


OFFSET = np.array([[0,0], [math.pi,0], [math.pi/2, math.pi/2]]) #locations of blue, green, red peaks 

class Halftone(BASE_Transform):

    def __init__(self, config: HalftoneConfig):
        # The Transform will apply the data augmentation to any data specified in its configuration's
        # ApplyTo field.
        # It may specify an additional list of keys and pass it to the parent constructor as the
        # required_transform_metadata parameter. These data items will always be retrieved and are used
        # for calculating the augmentation to apply to the ApplyTo data. These metadata items will not
        # be altered by the Transform
        # This example just requests the dataset index for the piece of data
        required_transform_metadata = ["index"]

        super().__init__(config, required_transform_metadata=required_transform_metadata)
        self.config: HalftoneConfig = cast(HalftoneConfig, config)
        xx, yy = np.mgrid[0:1600, 0:1600]
        scale = 2 * math.pi / self.config.Period
        xx = xx * scale
        yy = yy * scale

        signal = []
        for cc in [0, 1, 2]:
            signal.append(np.sin(xx + OFFSET[cc][0]) * np.sin(yy + OFFSET[cc][1]))
        signal = (np.dstack((signal[0], signal[1], signal[2])) + 1.0) * (255./2) # signal from 0..255
        signal_uint8 = signal.astype(np.uint8)
        self.halftonebase, _ = image_dtype_utils.transform_data_type(
            signal_uint8,
            to_type=image_dtype_utils.ImageDataType.PILImage
        )


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

        warped_signal = RandomPerspective(distortion_scale = 0.5, fill=128)(self.halftonebase)
        cropped_signal = CenterCrop((w,h))(warped_signal)
        warped_halftone_signal, _ = image_dtype_utils.transform_data_type(
            cropped_signal,
            to_type=image_dtype_utils.ImageDataType.NPArray
        )
        warped_halftone_float = warped_halftone_signal.astype(float) * (0.5 / 255) + 0.75 #signal from 0.75 to 1.25

        # 4. Perform the transformation of the data
        halftone_image = data_to_transform * warped_halftone_float[0:w, 0:h]

        # 5. Convert the transformed image back to the original Data type.
        #transformed_data, _ = image_dtype_utils.transform_data_type(grayscale_image, to_type=input_datatype,
        #                                                            device=device)
        transformed_data = np.clip(halftone_image, 0, 255).astype(np.uint8)
        additional_data = MLToolkitDictionary({"halftone_signal": warped_halftone_signal})

        return transformed_data, additional_data
