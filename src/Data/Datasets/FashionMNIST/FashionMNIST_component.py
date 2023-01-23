from typing import List, Optional, cast

from dneg_ml_toolkit.src.Data.Datasets.BASE_Dataset.BASE_Dataset_component import BASE_Dataset
from dneg_ml_toolkit.src.Data.Transforms.BASE_Transform.BASE_Transform_component import BASE_Transform
from dneg_ml_toolkit.src.Data.ml_toolkit_dictionary import MLToolkitDictionary
from torch.utils.data.dataset import T_co

from dneg_ml_toolkit.src.Data.Transforms.ToTensor.ToTensor_config import ToTensorConfig
from dneg_ml_toolkit.src.Data.Transforms.ToTensor.ToTensor_component import ToTensor

from src.Data.Datasets.FashionMNIST.FashionMNIST_config import FashionMNISTConfig

from torchvision.datasets import FashionMNIST as pytorch_FashionMNIST
import numpy as np


class FashionMNIST(BASE_Dataset):

    def __init__(self, config: FashionMNISTConfig, transforms: List[BASE_Transform]):

        # FashionMNIST returns PIL images, so append a ToTensor Transform to the end of the
        # configured Transforms so that the data is always returned in tensor format.
        transforms.append(ToTensor(ToTensorConfig(Type="ToTensor", ApplyTo=["data"])))

        super().__init__(config, allow_multiple_sources=False, transforms=transforms, check_source_exists=False)

        # This just informs the type checker that the config is of type FashionMNISTDatasetConfig,
        # since self.config is set in the parent's constructor, which is not aware of FashionMNISTDatasetConfig,
        # so is types self.config as the parent BASE_DatasetConfig.
        # It doesn't affect the code execution, but provides robustness through type checking, and
        # helps with code completion and access to documentation when using an IDE.
        self.config: FashionMNISTConfig = cast(FashionMNISTConfig, self.config)

        self.data_source = pytorch_FashionMNIST(root=config.Source, train=self.config.TrainingSet, download=True)

        self._image_resolution: Optional[List[int]] = None
        self._classes: Optional[List[int]] = None  # Private variable that is lazy-loaded in get_classes below

    def __len__(self) -> int:
        return len(self.data_source)

    def __getitem__(self, index) -> T_co:
        image, _ = self.data_source[index]

        train_dict = MLToolkitDictionary({"data": image, "index": index})
        target_dict = self.get_item_metadata(index)

        # Apply all configured Transforms
        train_dict, target_dict = self.apply_transforms(train_dict, target_dict)

        return train_dict, target_dict

    def get_item_metadata(self, index: int) -> MLToolkitDictionary:
        """
        Get the metadata for the provided index

        Args:
            index: Valid dataset index

        Returns:
            An MLToolkitDictionary of the metadata associated with the index
        """

        _, target = self.data_source[index]

        metadata = MLToolkitDictionary({"target": target, "index": index})

        return metadata

    def get_data_shape(self) -> List[int]:
        """
        Get the image resolution of the data in the dataset, assuming that all images have the same resolution.

        Returns:
            Image resolution of the image data after all Transforms have been applied, in [H,W,C] format
        """

        if self._image_resolution is None:
            data, _ = self.__getitem__(0)
            shape = np.array(data['data'].shape)  # Shape in C, H, W

            # Store the image resolution once it has been retrieved, so that future calls to this function
            # do not need to do any data processing
            self._image_resolution = np.roll(shape, -1).tolist()  # Swap dimensions to H, W, C

        return self._image_resolution

    def get_classes(self) -> List[int]:
        """
        Gets a list of all the unique classes in the dataset

        Returns:
            Unique class list
        """

        if self._classes is None:
            # Only loop through the full dataset once to get the classes, then store the result
            # and use it for all future calls to this function
            self._classes = []
            for i in range(len(self)):
                self._classes.append(self.get_item_metadata(i)["target"])

        unique_classes = list(set(self._classes))

        return unique_classes
