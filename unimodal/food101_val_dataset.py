from .base_dataset import BaseDataset
import torch
import random, os
import clip
import numpy as np

class FOOD101DatasetValidation(BaseDataset):
    def __init__(self, data_dir="", image_size=384, split="val", preprocess=None, **kwargs):
        assert split in ["val", "test"]
        self.split = split
        self.preprocess = preprocess

        if split == "val":
            names = ["Food101/food101_val"]
        elif split == "test":
            names = ["Food101/food101_test"] 

        super().__init__(
            data_dir=data_dir,
            image_size=image_size,
            preprocess=preprocess,
            names=names,
            text_column_name="text",
            remove_duplicate=False,
        )
        

    def __getitem__(self, index):
        image_index, question_index = self.index_mapper[index]
            
        image_tensor = self.get_image(index)["image"]
        text = self.get_text(index)["text"]
        labels = self.table["label"][image_index].as_py()
        
        return {
            "image": image_tensor,
            "text": text,
            "label": labels,
            "key": index,
        }
