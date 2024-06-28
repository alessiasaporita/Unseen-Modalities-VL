import torch
from unimodal.food101_val_dataset import FOOD101DatasetValidation
from unimodal.mmimdb_val_dataset import MMIMDBDatasetVal
from unimodal.hatememes_val_dataset import HateMemesDatasetVal
import unimodal.clip as clip
from utils.heads import MLP
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
from utils.my_metrics import AUROC
from utils.my_metrics import F1_Score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_path",
        type=str,
        default="/work/tesi_asaporita/UnseenModalities-VL/image/checkpoint/image_9.pt",
    )
    parser.add_argument(
        "--text_path",
        type=str,
        default="/work/tesi_asaporita/UnseenModalities-VL/text/checkpoint/text_13.pt",
    )
    parser.add_argument("--data-root", type=str, default='/work/tesi_asaporita/MissingModalities/datasets') 
    parser.add_argument(
        "--modality", type=str, help="image or text", default="image",
    ) 
    parser.add_argument(
        "--save_name", type=str, help="name to save the predictions", default="1e4",
    )
    parser.add_argument(
        "--num_classes", type=int, help="number of classes samples", default=23,
    )
    parser.add_argument("--dataset", default='mmimdb', type=str, choices=["mmimdb", "Hatefull_Memes", "Food101"])
    parser.add_argument("--image_size", default=384, type=int)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, preprocess = clip.load('ViT-B/32', device=device)
    head = MLP(512, args.num_classes).to(device)

    if args.modality=="image":
        checkpoint = torch.load(args.image_path)
        model.visual.load_state_dict(checkpoint["visual"])
        head.load_state_dict(checkpoint["head"])
    else:
        checkpoint = torch.load(args.text_path)
        head.load_state_dict(checkpoint["head"])
        for name, param in model.named_parameters():
            if name in checkpoint:
                param.data.copy_(checkpoint[name].data)

    model.eval()
    head.eval()

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
            modality=args.modality,
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
        with tqdm.tqdm(total=len(dataset)) as pbar:
            for i, sample in enumerate(test_dataloader):
                image = sample['image'][0].to(device, non_blocking=True) #(1, 3, 384, 384)
                text = sample['text']
                text_embedding = clip.tokenize(
                    text, truncate=True).to(device, non_blocking=True) #(1, 77)
                labels = sample['label'] #(B, )
                
                with torch.no_grad(): 
                    with torch.cuda.amp.autocast():
                        if args.modality=='image':
                            features = model.encode_image(image)
                        else:
                            features = model.encode_text(text_embedding)
                        outputs = head(features) 
                
                if args.dataset == 'Food101':
                    outputs = torch.softmax(outputs, dim=-1) #(B, 3086)
                    predictions = outputs.detach().cpu().numpy()
                    labels = labels.numpy()[0]

                    if np.argmax(predictions) == labels:
                        acc += 1
                    print(i+1, '/', num_of_samples, 'Accuracy:', acc / (i+1))
                    metric = acc / (i+1)
                elif args.dataset == 'Hatefull_Memes':
                    auroc.update(outputs.float(), labels)
                    aur = auroc.compute(use_softmax=True)
                    metric = aur.item()
                else:#mmimdb
                    f1_metric.update(outputs.float(), labels) 
                    F1_Micro, F1_Macro, F1_Samples, F1_Weighted = f1_metric.compute()
                    metric = F1_Macro.item()

                pbar.update()

                f.write(
                    "{}/{}, Metric:, {}\n".format(
                        i+1,
                        num_of_samples,
                        metric,
                    )
                        )
                f.flush()
    f.close()
    
            