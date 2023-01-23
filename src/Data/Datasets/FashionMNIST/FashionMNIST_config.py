from dataclasses import dataclass

from dneg_ml_toolkit.src.Data.Datasets.BASE_Dataset.BASE_Dataset_config import BASE_DatasetConfig


@dataclass
class FashionMNISTConfig(BASE_DatasetConfig):
    TrainingSet: bool = True
