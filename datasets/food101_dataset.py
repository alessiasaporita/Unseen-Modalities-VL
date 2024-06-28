from .base_dataset import BaseDataset
import torch
import random, os
import clip
import numpy as np

class FOOD101Dataset(BaseDataset):
    def __init__(self, *args, split="", preprocess=None, missing_info={}, **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split
        self.preprocess = preprocess

        if split == "train":
            names = ["Food101/food101_train"]
        elif split == "val":
            names = ["Food101/food101_val"]
        elif split == "test":
            names = ["Food101/food101_test"] 

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
            
            if missing_ratio > 0 and split == "train":
                missing_index = random.sample(range(total_num), int(total_num*missing_ratio)) #61127

                if missing_type == 'text':
                    missing_table[missing_index] = 1
                elif missing_type == 'image':
                    missing_table[missing_index] = 2
                elif missing_type == 'both':
                    missing_table[missing_index] = 1 #image indices
                    missing_index_image  = random.sample(missing_index, int(len(missing_index)*both_ratio)) #30563
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
                image_pseudo = torch.zeros((101,))
            else:
                image_mask = np.ones((512, 1))
                image_pseudo_path = "/work/tesi_asaporita/UnseenModalities-VL/Food101/image/image_pseudo/" + str(index) + ".npy"
                if os.path.exists(image_pseudo_path):
                    image_pseudo = torch.Tensor(np.load(image_pseudo_path)) #(N, 101)
                    image_pseudo = torch.mean(image_pseudo[-10:], dim=0)
                else:
                    image_pseudo = torch.zeros((101,))
   
            #missing text, dummy text is ''
            if self.missing_table[image_index] == 1:
                text = ''
                text_mask = np.zeros((512, 1))
                text_pseudo = torch.zeros((101,))
            else:
                text_mask = np.ones((512, 1))
                text_pseudo_path = "/work/tesi_asaporita/UnseenModalities-VL/Food101/text/text_pseudo/" + str(index) + ".npy"
                if os.path.exists(text_pseudo_path):
                    text_pseudo = torch.Tensor(np.load(text_pseudo_path)) #(N, 101)
                    text_pseudo = torch.mean(text_pseudo[-10:], dim=0) #(101, )
                else:
                    text_pseudo = torch.zeros((101,))
        else:
            text_mask = np.ones((512, 1))
            text_pseudo = torch.zeros((101,))
            image_mask = np.ones((512, 1))
            image_pseudo = torch.zeros((101,))
        
        labels = self.table["label"][image_index].as_py()
        
        return( 
            image_tensor,
            text,
            labels,
            image_mask.astype(np.float32),
            text_mask.astype(np.float32),
            self.missing_table[image_index].item(),
            index,
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
                image_tensor = torch.ones(image_tensor[idx].size()).float()
            
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
            "missing_type": self.missing_table[image_index].item(),
        }
