from datasets import FOOD101Dataset
from .datamodule_base import BaseDataLoader
from collections import defaultdict


class FOOD101DataModule(BaseDataLoader):
    def __init__(self, preprocess_train, preprocess_val, preprocess_test, *args, **kwargs):
        super().__init__(preprocess_train, preprocess_val, preprocess_test, *args, **kwargs)

    @property
    def dataset_cls(self):
        return FOOD101Dataset

    @property
    def dataset_name(self):
        return "food101"

    def setup(self):
        super().setup()

