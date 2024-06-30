import torch
from unimodal.food101_val_dataset import FOOD101DatasetValidation
from unimodal.hatememes_val_dataset import HateMemesDatasetVal
from unimodal.mmimdb_val_dataset import MMIMDBDatasetVal
import pdb
import torch
import argparse
import tqdm
import os
import numpy as np
import torch.nn as nn
import random
import warnings
import torch.nn.functional as F
import datetime
import clip
from vit import ViT
from feature_reorganization import ViTReorganization
from utils.my_metrics import F1_Score, AUROC



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_position",
        type=int,
        help="number of latent tokens",
        default=512,
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="/work/tesi_asaporita/checkpoint/dataset/image/checkpoint/image_9.pt",
    )
    parser.add_argument(
        "--text_path",
        type=str,
        default="/work/tesi_asaporita/checkpoint/dataset/text/checkpoint/text_13.pt",
    )
    parser.add_argument("--dataset", default='Food101', type=str, choices=["mmimdb", "Hatefull_Memes", "Food101"])
    parser.add_argument("--finetuning", type=bool, default=False) 
    parser.add_argument("--data-root", type=str, default='/work/tesi_asaporita/datasets') 
    parser.add_argument(
        "--save_name", type=str, help="name to save the predictions", default="1e4",
    )
    parser.add_argument(
        "--resume_checkpoint", type=str, help="path to the checkpoint file", default="checkpoint/best_multimodal_KL",
    )
    parser.add_argument(
        "--num_classes", type=int, help="number of classes samples", default=101,
    )
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    model, preprocess = clip.load('ViT-B/32', device=device)
    if args.finetuning:
        checkpoint = torch.load(args.image_path)
        model.visual.load_state_dict(checkpoint["visual"])

        checkpoint = torch.load(args.text_path)
        for name, param in model.named_parameters():
            if name in checkpoint:
                param.data.copy_(checkpoint[name].data)
    model.eval()

    multimodal_model = ViT(num_classes = args.num_classes, dim = 256, depth = 6, heads = 8, mlp_dim = 512, num_position = args.num_position)
    multimodal_model = torch.nn.DataParallel(multimodal_model)
    multimodal_model = multimodal_model.to(device)

    reorganization_module = ViTReorganization(dim = 256, depth = 6, heads = 8, mlp_dim = 512, num_position = args.num_position)
    reorganization_module = torch.nn.DataParallel(reorganization_module)
    reorganization_module = reorganization_module.to(device)

    checkpoint = torch.load(args.resume_checkpoint)
    multimodal_model.load_state_dict(checkpoint['model'])
    multimodal_model.eval()

    reorganization_module.load_state_dict(checkpoint['reorganization'])
    reorganization_module.eval()

    if args.dataset == 'Hatefull_Memes':
        dataset = HateMemesDatasetVal(
            data_dir=args.data_root,
            image_size=384,
            split="test",
            preprocess=preprocess,
        )
    elif args.dataset == 'Food101':
        dataset = FOOD101DatasetValidation(
            data_dir=args.data_root,
            image_size=384,
            split="test",
            preprocess=preprocess,
        )
    else:
        dataset = MMIMDBDatasetVal(
            data_dir=args.data_root,
            image_size=384,
            modality='late_fusion',
            split="test",
            preprocess=preprocess,
        )

    test_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=4,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    num_of_samples = len(test_dataloader)
    acc = 0
    f1_metric = F1_Score()
    auroc = AUROC()
    save_path = 'predictions/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    pred_path = "predictions/{}.csv".format(args.save_name)

    with open(pred_path, "a") as f:
        for i, sample in enumerate(test_dataloader):
            image = sample['image'][0].to(device, non_blocking=True) #(1, 3, 384, 384)
            text = sample['text']
            text_embedding = clip.tokenize(
                text, truncate=True).to(device, non_blocking=True) #(B, 77)
            labels = sample['label'] #(B, )

            text_mask = torch.tensor(np.ones((512, 1)).astype(np.float32)).cuda()
            image_mask = torch.tensor(np.ones((512, 1)).astype(np.float32)).cuda()

            with torch.no_grad(): 
                with torch.cuda.amp.autocast():
                    image_features = model.encode_image(image) #(B, 50, 512)
                    text_features = model.encode_text(text_embedding) #(B, 77, 512)
                    image, text = reorganization_module(image_features, text_features) #(B, 512, 256) 
                    outputs = multimodal_model(image, text, image_mask, text_mask)  #(B, 2, 101)   
            
            if args.dataset == 'Food101':
                outputs = torch.softmax(outputs, dim=-1)                
                outputs = torch.mean(outputs, dim=1) #(B, 101)          
                predictions = outputs.detach().cpu().numpy()
                labels = labels.numpy()[0]

                if np.argmax(predictions) == labels:
                    acc += 1
                print(i+1, '/', num_of_samples, 'Accuracy:', acc / (i+1))
                metric = acc / (i+1)

            elif args.dataset == 'Hatefull_Memes':
                outputs = torch.softmax(outputs, dim=-1)
                outputs = torch.mean(outputs, dim=1)
                auroc.update(outputs, labels)
                aur = auroc.compute(use_softmax=False)
                metric = aur.item()
                
            else:#mmimdb
                outputs = torch.mean(outputs, dim=1)
                f1_metric.update(outputs, labels) 
                F1_Micro, F1_Macro, F1_Samples, F1_Weighted = f1_metric.compute()
                metric = F1_Macro.item()
            f.write(
                "{}/{}, Metric:, {}\n".format(
                    i+1,
                    num_of_samples,
                    metric,
                )
                    )
            f.flush()
    f.close()
    
            