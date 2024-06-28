from datasets import HateMemesDataset
from .datamodule_base import BaseDataLoader
from collections import defaultdict


class HateMemesDataModule(BaseDataLoader):
    def __init__(self, preprocess_train, preprocess_val, preprocess_test, *args, **kwargs):
        super().__init__(preprocess_train, preprocess_val, preprocess_test, *args, **kwargs)

    @property
    def dataset_cls(self):
        return HateMemesDataset

    @property
    def dataset_name(self):
        return "Hatefull_Memes"

    def setup(self):
        super().setup()
