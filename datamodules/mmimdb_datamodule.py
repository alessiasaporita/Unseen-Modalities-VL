from datasets import MMIMDBDataset
from .datamodule_base import BaseDataLoader


class MMIMDBDataModule(BaseDataLoader):
    def __init__(self, preprocess_train, preprocess_val, preprocess_test, *args, **kwargs):
        super().__init__(preprocess_train, preprocess_val, preprocess_test, *args, **kwargs)

    @property
    def dataset_cls(self):
        return MMIMDBDataset

    @property
    def dataset_name(self):
        return "mmimdb"

    def setup(self):
        super().setup()

