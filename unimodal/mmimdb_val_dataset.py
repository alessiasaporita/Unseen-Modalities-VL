from .base_dataset import BaseDataset
import torch
import random
import os
import clip

class MMIMDBDatasetVal(BaseDataset):
    def __init__(self, data_dir="", modality=None, image_size=384, split="val", preprocess=None, **kwargs):
        assert split in ["val", "test"]
        self.split = split
        self.preprocess = preprocess
        self.modality = modality

        if modality == 'text' or modality == 'late_fusion': #multiple plots for the same image
            if split == "val":
                names = ["mmimdb/mmimdb_dev"]
            elif split == "test":
                names = ["mmimdb/mmimdb_test"]
        elif modality == 'image': #single plot for each image-> unique dict entries
            if split == "val":
                names = ["mmimdb/mmimdb_dev_single_plot"]
            elif split == "test":
                names = ["mmimdb/mmimdb_test_single_plot"]
        else:
            raise NotImplementedError


        super().__init__(
            data_dir=data_dir,
            image_size=image_size,
            preprocess=preprocess,
            names=names,
            text_column_name="plots",
            remove_duplicate=False,
        )
        

    def __getitem__(self, index):
        
        image_index, question_index = self.index_mapper[index]

        image_tensor = self.get_image(index)["image"]
        text = self.get_text(index)["text"] 
        labels = torch.tensor(self.table["label"][image_index].as_py())

        return {
            "image": image_tensor,
            "text": text,
            "label": labels,
            "key": index,
        }
