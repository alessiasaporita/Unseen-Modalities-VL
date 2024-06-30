from .base_dataset import BaseDataset
import torch
import random, os
import clip
import numpy as np

class HateMemesDataset(BaseDataset):
    def __init__(self, *args, split="", preprocess=None, missing_info={}, **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split
        self.preprocess = preprocess

        if split == "train":
            names = ["Hatefull_Memes/hatememes_train"]
        elif split == "val":
            names = ["Hatefull_Memes/hatememes_dev"]
        elif split == "test":
            names = ["Hatefull_Memes/hatememes_test"] 

        super().__init__(
            *args,
            **kwargs,
            preprocess=preprocess,
            names=names,
            text_column_name="text",
            remove_duplicate=False,
        )
        
        # missing modality control        
        self.simulate_missing = missing_info['simulate_missing']
        missing_ratio = missing_info['ratio'][split]
        mratio = str(missing_ratio).replace('.','')
        missing_type = missing_info['type'][split]    
        both_ratio = missing_info['both_ratio']
        missing_table_root = missing_info['missing_table_root']
        missing_table_name = f'{names[0]}_missing_{missing_type}_{mratio}.pt'
        missing_table_path = os.path.join(missing_table_root, missing_table_name)
        
        # use image data to formulate missing table
        total_num = len(self.table['image'])
        
        if os.path.exists(missing_table_path):
            missing_table = torch.load(missing_table_path)
            if len(missing_table) != total_num:
                print('missing table mismatched!')
                exit()
        else:
            missing_table = torch.zeros(total_num)
            
            if missing_ratio > 0:
                missing_index = random.sample(range(total_num), int(total_num*missing_ratio))

                if missing_type == 'text':
                    missing_table[missing_index] = 1
                elif missing_type == 'image':
                    missing_table[missing_index] = 2
                elif missing_type == 'both':
                    missing_table[missing_index] = 1 #img indices
                    missing_index_image  = random.sample(missing_index, int(len(missing_index)*both_ratio))
                    missing_table[missing_index_image] = 2
                    
                torch.save(missing_table, missing_table_path)

        self.missing_table = missing_table

    def get_value_by_index(self, index):
        image_index, question_index = self.index_mapper[index]
        
        image_tensor = self.get_image(index)["image"]
        text = self.get_text(index)["text"]

        if self.split=='train':
            # missing image, dummy image is all-one image
            if self.missing_table[image_index] == 2:
                for idx in range(len(image_tensor)):
                    image_tensor[idx] = torch.ones(image_tensor[idx].size()).float()
                image_mask = np.zeros((512, 1))
                image_pseudo = torch.zeros((2,))
            else:
                image_mask = np.ones((512, 1))
                image_pseudo_path = "/work/tesi_asaporita/checkpoint/Hatefull_Memes/image/image_pseudo2/" + str(index) + ".npy"
                if os.path.exists(image_pseudo_path):
                    image_pseudo = torch.Tensor(np.load(image_pseudo_path)) #(N, 2)
                    image_pseudo = torch.mean(image_pseudo[-10:], dim=0)
                else:
                    image_pseudo = torch.zeros((2,))

            #missing text, dummy text is ''
            if self.missing_table[image_index] == 1:
                text = ''
                text_mask = np.zeros((512, 1))
                text_pseudo = torch.zeros((2,))
            else:
                text_mask = np.ones((512, 1))
                text_pseudo_path = "/work/tesi_asaporita/checkpoint/Hatefull_Memes/text/text_pseudo2/" + str(index) + ".npy"
                if os.path.exists(text_pseudo_path):
                    try:
                        text_pseudo = torch.Tensor(np.load(text_pseudo_path)) #(N, 2)
                        text_pseudo = torch.mean(text_pseudo[-10:], dim=0) #(2, )
                    except Exception as e:
                        print(f"Error: Problem with file at {text_pseudo_path}")
                else:
                        text_pseudo = torch.zeros((2,))
        else:
            text_mask = np.ones((512, 1))
            text_pseudo = torch.zeros((2,))
            image_mask = np.ones((512, 1))
            image_pseudo = torch.zeros((2,))
        
        labels = self.table["label"][image_index].as_py()
        
        return( 
            image_tensor,
            text,
            labels,
            image_mask.astype(np.float32),
            text_mask.astype(np.float32),
            self.missing_table[image_index].item(),
            image_index,
            image_pseudo,
            text_pseudo,
        )

    def __getitem__(self, index):
        # index -> pair data index
        # image_index -> image index in table
        # question_index -> plot index in texts of the given image
        image_index, question_index = self.index_mapper[index]
        
        if self.split == 'train' and self.missing_table[image_index] == 0:
            simulate_missing_type = random.choice([0,1,2])
            
        image_tensor = self.get_image(index)["image"]
        
        # missing image, dummy image is all-one image
        if self.missing_table[image_index] == 2:
            for idx in range(len(image_tensor)):
                image_tensor[idx] = torch.ones(image_tensor[idx].size()).float()
            
        #missing text, dummy text is ''
        if self.missing_table[image_index] == 1:
            text = ''
        else:
            text = self.get_text(index)["text"]

        
        labels = self.table["label"][image_index].as_py()
        
        return {
            "image": image_tensor,
            "text": text,
            "label": labels,
            "missing_type": self.missing_table[image_index].item()+simulate_missing_type,
        }
