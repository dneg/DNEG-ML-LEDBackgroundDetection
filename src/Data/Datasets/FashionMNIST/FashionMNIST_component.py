from typing import List, Optional, cast, Tuple

from dneg_ml_toolkit.src.Data.Datasets.BASE_Dataset.BASE_Dataset_component import BASE_Dataset
from dneg_ml_toolkit.src.Data.Transforms.BASE_Transform.BASE_Transform_component import BASE_Transform
from dneg_ml_toolkit.src.Data.ml_toolkit_dictionary import MLToolkitDictionary

from dneg_ml_toolkit.src.Data.Transforms.ToTensor.ToTensor_config import ToTensorConfig
from dneg_ml_toolkit.src.Data.Transforms.ToTensor.ToTensor_component import ToTensor

from src.Data.Datasets.FashionMNIST.FashionMNIST_config import FashionMNISTConfig

from torchvision.datasets import FashionMNIST as pytorch_FashionMNIST
import numpy as np


class FashionMNIST(BASE_Dataset):
    """
    Dataset Component to connect to the PyTorch FashionMNIST dataset and serve its images to the Dataloader
    """

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

    def __getitem__(self, index) -> Tuple[MLToolkitDictionary, MLToolkitDictionary]:
        """
        Core Dataset function that accepts a data index, retrieves the data at that index, its corresponding
        metadata/targets, passes the item through the Transform pipeline, and returns the item to the calling Dataloader
        Args:
            index: Integer index of the data item within the Dataset

        Returns:
            A tuple of Toolkit dictionaries, the first containing the data to send to the Network, the second with the
            metadata/targets to send to the Loss.
        """
        image, _ = self.data_source[index]

        # All data is transported through ML Toolkit systems in ML Toolkit dictionaries
        # (a custom dictionary for holding Tensors). This provides flexibility for training,
        # as multiple tensors can be passed into the forward pass of the Network and Loss at the same time.
        # The ML Toolkit standard is for the Dataset to store the core tensor, such as the image in this case,
        # under the "data" keyword, and the ground truth under the "target" keyword.
        train_dict = MLToolkitDictionary({"data": image, "index": index})
        target_dict = self.get_item_metadata(index)

        # Pass all the data for the sample through the Transform pipeline
        train_dict, target_dict = self.apply_transforms(train_dict, target_dict)

        return train_dict, target_dict

    def get_item_metadata(self, index: int) -> MLToolkitDictionary:
        """
        Get the metadata for the provided index. FashionMNIST stores the data and the metadata together,
        but the standard for ML Toolkit datasets is to store the data and metadata separately, so that
        analysis on the metadata can be performed without having to load all the data into memory.

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
        Get the image resolution of the data in the dataset, assuming that all images have the same resolution. This
        is passed to the Network and can be used to drive the creation of the Network architecture

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
        Gets a list of all the unique classes in the dataset. See the ClassificationTrainModule where this is used
        to dynamically configure the Network's output shapes based on the number of classes in the dataset.

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
