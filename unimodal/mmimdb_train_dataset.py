from .base_dataset import BaseDataset
import torch
import random
import os
import clip
import csv


class MMIMDBDatasetTrain(BaseDataset):
    def __init__(self, data_dir="", image_size=384, split="train", preprocess=None, modality=None, **kwargs):
        assert split in ["train", "val", "test"]
        self.preprocess = preprocess
        self.modality = modality

        if modality == 'text': #multiple plots
            names = ["mmimdb/mmimdb_train"]
        else: #single image
            names = ["mmimdb/mmimdb_train_single_plot"]
           
        samples=[]
        if modality=='image': #7776 unique images samples
            with open("annotations/imdb_annotations/image_samples.csv") as f: #single image
                f_csv = csv.reader(f)
                for i, row in enumerate(f_csv):
                    samples.append(int(row[0]))
            f.close()
        else: #16170 different plots samples
            with open("annotations/imdb_annotations/text_indices.csv") as f: #multiple plots
                f_csv = csv.reader(f)
                for i, row in enumerate(f_csv):
                    samples.append(int(row[0]))
            f.close()
        
        self.data = samples

        super().__init__(
            data_dir=data_dir,
            image_size=image_size,
            preprocess=preprocess,
            names=names,
            text_column_name="plots",
            remove_duplicate=False,
        )
        
    def __len__(self): #16108 images, 16170 texts = 32278
        return len(self.data)

    def __getitem__(self, index):
        index = self.data[index]
        image_index, question_index = self.index_mapper[index]
    
        if self.modality == 'text':   
            text = self.get_text(index)["text"]
            image_tensor = ''
        else:
            image_tensor = self.get_image(index)["image"]
            text = ''
        
        labels = torch.tensor(self.table["label"][image_index].as_py())
        
        return {
            "image": image_tensor,
            "text": text,
            "label": labels,
            "key": index,
        }
