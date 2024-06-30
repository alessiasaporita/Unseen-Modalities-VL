
import webdataset as wds
from tqdm import tqdm
import random
import json
import argparse
from PIL import Image, ImageFilter
import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio
import random
import numpy as np
import csv
import os
import torch.nn as nn
import clip
from datasets.food101_dataset import FOOD101Dataset
from datasets.hatememes_dataset import HateMemesDataset
from datasets.mmimdb_dataset import MMIMDBDataset
from datamodules.multitask_datamodule import MTDataModule


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='standard')
    parser.add_argument('--split', type=str, default='training')
    parser.add_argument('--job_idx', type=int, default=0)
    parser.add_argument('--job_size', type=int, default=5000)
    parser.add_argument("--data-root", type=str, default='/work/tesi_asaporita/datasets')
    parser.add_argument("--clip-model", default='ViT-B/32', type=str, choices=['RN50x16', 'ViT-B/32'])
    parser.add_argument("--num_workers", default=16, type=int)
    parser.add_argument("--batch-size", default=1, type=int) 
    parser.add_argument('--image_path', type=str, default='')
    parser.add_argument('--text_path', type=str, default='')
    parser.add_argument('--finetuning', type=bool, default=False)
    
    
    # missing modality config
    parser.add_argument("--missing-ratio", default=1, type=float)
    parser.add_argument("--missing-type", default='both', choices=['text', 'image', 'both'], type=str)
    parser.add_argument("--both-ratio", default=0.5, type=int)
    parser.add_argument("--missing-table-root", default='/work/tesi_asaporita/datasets/missing_tables/', type=str)
    parser.add_argument("--simulate-missing", default=False, type=bool)
    parser.add_argument("--dataset", default='Food101', type=str, choices=["mmimdb", "Hatefull_Memes", "Food101"])
    
    # Image setting
    parser.add_argument("--image-size", default=384, type=int)
    parser.add_argument("--max-image-len", default=-1, type=int)
    parser.add_argument("--draw-false-image", default=1, type=int)
    parser.add_argument("--image-only", default=False, type=bool)

    # Text Setting
    parser.add_argument("--vqav2-label-size", default=3129, type=int)
    parser.add_argument("--max-text-len", default=77, type=int)

    parser.add_argument("--test-ratio", default=None, type=int)
    parser.add_argument("--test-type", default=None, type=str)

    args = parser.parse_args()
    print(args)

    shard_start = args.job_idx * args.job_size
    shard_end = shard_start + args.job_size
    split = args.split

    pattern = '/work/tesi_asaporita/webdataset_no_finetuning/food101-' + split + '-%03d.tar' % args.job_idx

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    model, preprocess = clip.load(args.clip_model, device=device)
    if args.finetuning:
        checkpoint = torch.load(args.image_path, map_location=torch.device('cpu')) 
        model.visual.load_state_dict(checkpoint["visual"])
        checkpoint = torch.load(args.text_path, map_location=torch.device('cpu')) 
        for name, param in model.named_parameters():
            if name in checkpoint:
                param.data.copy_(checkpoint[name].data)
    model.eval()

    missing_info = {
        'ratio' : {'train': args.missing_ratio, 'val': args.missing_ratio, 'test': args.missing_ratio}, 
        'type' : {'train': args.missing_type, 'val': args.missing_type, 'test': args.missing_type},
        'both_ratio' : args.both_ratio,
        'missing_table_root': args.missing_table_root,
        'simulate_missing' : args.simulate_missing
    }
    if args.dataset=='Food101':  
        train_dataset = FOOD101Dataset(
            data_dir=args.data_root,
            image_size=args.image_size,
            split="train",
            preprocess=preprocess,
            missing_info=missing_info, 
        )
        val_dataset = FOOD101Dataset(
            data_dir=args.data_root,
            image_size=args.image_size,
            split="val",
            preprocess=preprocess, 
            missing_info=missing_info,
        )
    elif args.dataset=='mmimdb':
        train_dataset = MMIMDBDataset(
            data_dir=args.data_root,
            image_size=args.image_size,
            split="train",
            preprocess=preprocess, 
            missing_info=missing_info,
        )
        val_dataset = MMIMDBDataset(
            data_dir=args.data_root,
            image_size=args.image_size,
            split="val",
            preprocess=preprocess,
            missing_info=missing_info,
        )
    elif args.dataset=='Hatefull_Memes':
        train_dataset = HateMemesDataset(
            data_dir=args.data_root,
            image_size=args.image_size,
            split="train",
            preprocess=preprocess, 
            missing_info=missing_info,
        )
        val_dataset = HateMemesDataset(
            data_dir=args.data_root,
            image_size=args.image_size,
            split="val",
            preprocess=preprocess,
            missing_info=missing_info,
        )

    #dm = MTDataModule(preprocess, preprocess, preprocess, args)  
    #dm.setup()
    
    #train_dataset = dm.train_dataloader()
    #val_dataset = dm.val_dataloader()


    if split == 'training':
        dataset = train_dataset
        split_i = 0
    elif split == 'validation':
        dataset = val_dataset
        split_i = 1
    else:
        split_i = 2

    indexes = list(range(len(dataset))) #(0, len(dataset)-1)
    if split_i == 0 or split_i == 1: #shuffle train and validation
        random.Random(4).shuffle(indexes)

    
    dst = wds.TarWriter(pattern) 
    inserted = -1 
    for idx in tqdm(indexes):
        inserted += 1 
        if inserted >= shard_end:
            print("end my shard")
            break
        elif inserted < shard_end and inserted >= shard_start:
            image, text, label, image_mask, text_mask, missing_type, image_index, image_pseudo, text_pseudo = dataset.datasets[0].get_value_by_index(idx)
            """ 
            #missing_type = 1 = text missing = empty string
            #missing_type = 2 = image missing = image of ones
            #missing_type = 0 = modality complete
            """
            image = image[0].to(device, non_blocking=True) #(3, 384, 384)
            text_embedding = clip.tokenize(
                text, truncate=True).to(device, non_blocking=True) #(1, 77)
                
            with torch.no_grad():
                image_features = model.encode_image(image.unsqueeze(0)) #(1, 50, 512)
                text_features = model.encode_text(text_embedding) #(1, 77, 512)
            
            #image_features = image_features / image_features.norm(dim=1, keepdim=True)
            #text_features = text_features / text_features.norm(dim=1, keepdim=True)

            if split == 'training':
                sample = {
                    '__key__': str(idx),
                    'image_features.pth': image_features.squeeze(0), #tensor (50, 512) of float16
                    'image_mask.pth': torch.tensor(image_mask), #(512, 1)
                    'image_pseudo.pth': image_pseudo, #(n_classes, )
                    'text_features.pth': text_features.squeeze(0), #tensor (77, 512) of float16
                    'text_mask.pth': torch.tensor(text_mask), #(512, 1)
                    'text_pseudo.pth': text_pseudo, #(n_classes, )
                    'label.id': label, #()
                }
            else: #validation
                sample = {
                    '__key__': str(idx),
                    'image_features.pth': image_features.squeeze(0), #tensor (50, 512) of float16
                    'image_mask.pth': torch.tensor(image_mask),
                    'text_features.pth': text_features.squeeze(0), #tensor (77, 512) of float16
                    'text_mask.pth': torch.tensor(text_mask),
                    'label.id': label,
                }
            dst.write(sample)
    dst.close()




