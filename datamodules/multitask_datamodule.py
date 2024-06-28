import functools
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data.distributed import DistributedSampler

from . import _datamodules


class MTDataModule():
    def __init__(self, preprocess_train, preprocess_val, preprocess_test, args, dist=False):
        datamodule_keys = []
        datamodule_keys.append(args.dataset)
        
        assert len(datamodule_keys) > 0

        super().__init__()

        self.dm_keys = datamodule_keys
        self.dm_dicts = {key: _datamodules[key](preprocess_train, preprocess_val, preprocess_test, args) for key in datamodule_keys}
        self.dms = [v for k, v in self.dm_dicts.items()] #lista datamodules inizializzati
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.dist = dist

    def prepare_data(self):
        for dm in self.dms:
            dm.prepare_data()

    def setup(self):
        for dm in self.dms:
            dm.setup()

        self.train_dataset = ConcatDataset([dm.train_dataset for dm in self.dms])
        self.val_dataset = ConcatDataset([dm.val_dataset for dm in self.dms])
        self.test_dataset = ConcatDataset([dm.test_dataset for dm in self.dms])
        #self.tokenizer = self.dms[0].tokenizer
        
        self.collate = functools.partial(
            self.dms[0].train_dataset.collate)
        
        if self.dist:
            self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
            self.val_sampler = DistributedSampler(self.val_dataset, shuffle=True)
            self.test_sampler = DistributedSampler(self.test_dataset, shuffle=False)
        else:
            self.train_sampler = None
            self.val_sampler = None
            self.test_sampler = None
            
    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate,
            shuffle=True,
        )
        #return loader
        return self.train_dataset

    def val_dataloader(self, batch_size=None):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            sampler=self.val_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate,
            shuffle=True,
        )
        #return loader
        return self.val_dataset

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            sampler=self.test_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate,
            shuffle=False,
        )
        return loader
