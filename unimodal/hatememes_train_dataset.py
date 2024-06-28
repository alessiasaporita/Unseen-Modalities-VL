from .base_dataset import BaseDataset
import torch
import random, os
import clip
import csv

class HateMemesDatasetTrain(BaseDataset):
    def __init__(self, data_dir="", image_size=384, split="train", preprocess=None, modality=None, **kwargs):
        self.preprocess = preprocess
        self.modality = modality

        names = ["Hatefull_Memes/hatememes_train"]

        samples=[]
        if modality=='image': #4250 samples
            with open("annotations/hm_annotations/image_samples.csv") as f:
                f_csv = csv.reader(f)
                for i, row in enumerate(f_csv):
                    samples.append(int(row[0]))
            f.close()
        else: #4250 samples
            with open("annotations/hm_annotations/text_samples.csv") as f:
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
            text_column_name="text",
            remove_duplicate=False,
        )
        
    
    def __len__(self):
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
        
        labels = self.table["label"][image_index].as_py()
        
        return {
            "image": image_tensor,
            "text": text,
            "label": labels,
            "key": index,
        }
