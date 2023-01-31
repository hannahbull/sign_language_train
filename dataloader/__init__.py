from .features_dataloader import FeaturesDataset
from .features_dataloader_dense import DenseFeaturesDataset

dataset_dict = {
    'features': FeaturesDataset,
    'densefeatures': DenseFeaturesDataset,

}

__all__ = ['dataset_dict']