import torch
import argparse
import tqdm
import os, io, sys
import webdataset as wds
import shutil
import numpy as np
import torch.nn as nn
import random
from braceexpand import braceexpand
import warnings
from torch.cuda.amp import GradScaler
import torch.nn.functional as F
from vit import ViT
from feature_reorganization import ViTReorganization
import wandb
import math
import os
from torch.optim.lr_scheduler import MultiStepLR
from utils.my_metrics import AUROC
from utils.my_metrics import F1_Score

import torch.nn.functional as F

def dict_to_cuda(data):
    for key, value in data.items():
        data[key] = value.cuda()
    return data


class LabelSmoothLoss(nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing #0.1

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1) #class probabilities
        #tensor of the same shape as the input with values equal to self.smoothing / (input.size(-1) - 1.0)
        weight = input.new_ones(input.size()) * self.smoothing / (input.size(-1) - 1.0)
        #For the target class, it sets the value to 1.0 - self.smoothing.
        weight.scatter_(-1, target.unsqueeze(-1), (1.0 - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean() #cross-entropy loss
        return loss


def train_one_step(
    data,
    labels,
    masks,
    multimodal_model,
    reorganization_module,
    optim,
    loss_fn,
    scaler,
    indice,
    last_indice,
    gc,
    dataset, 
):
    image, text = reorganization_module( #feature projection = (B, 512, 256) = (B, k*, d*)
        data['Image'], data['Text'] 
    ) 

    outputs = multimodal_model( #(B, 2, num_classes), the two CLS tokens
            image, text, masks['Image'], masks['Text']
        ) 
    if dataset=='mmimdb':
        labels=labels.float()
        output_loss = loss = (F.binary_cross_entropy_with_logits(outputs[:,0], labels) + F.binary_cross_entropy_with_logits(outputs[:,1], labels)) * 0.5 
    else:
        output_loss = loss = (loss_fn(outputs[:,0], labels) + loss_fn(outputs[:,1], labels)) * 0.5 
    loss = loss / gc
    scaler.scale(loss).backward()

    if((indice + 1) % gc == 0) or (indice + 1 == last_indice):
        scaler.step(optim)
        scaler.update()
        optim.zero_grad()

    return outputs, output_loss


def val_one_step(
    data,
    labels,
    masks,
    multimodal_model,
    reorganization_module,
    loss_fn,
    gc,
    dataset,
):
    with torch.no_grad():
        image, text = reorganization_module(
            data['Image'], data['Text']
        )
        outputs = multimodal_model(
            image, text, masks['Image'], masks['Text']
        )
        if dataset=='mmimdb':
            labels=labels.float()
            output_loss = loss = (F.binary_cross_entropy_with_logits(outputs[:,0], labels) + F.binary_cross_entropy_with_logits(outputs[:,1], labels)) * 0.5 
        else:
            output_loss = loss = (loss_fn(outputs[:,0], labels) + loss_fn(outputs[:,1], labels)) * 0.5 

    return outputs, output_loss


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lr", type=float, help="learning rate", default=1e-1
    )  
    parser.add_argument("--batch_size", type=int, help="batch size", default=96)
    parser.add_argument(
        "--train_data_path",
        type=str,
        help="path to train data",
        default="/work/tesi_asaporita/webdataset/mmimdbtraining-{000..010}.tar",
    )
    parser.add_argument(
        "--val_data_path",
        type=str,
        help="path to validation data",
        default="/work/tesi_asaporita/webdataset/mmimdb-validation-{000..003}.tar",
    )
    parser.add_argument(
        "--n_train_samples", type=int, help="number of training samples", default=32278,
    )
    parser.add_argument(
        "--n_val_samples", type=int, help="number of training samples", default=5411,
    )
    parser.add_argument(
        "--dataset", type=str, default='mmimdb',
    )
    parser.add_argument(
        "--save_name", type=str, help="name to save the model", default="1e-1",
    ) 
    parser.add_argument(
        "--resume_training", type=bool, help="resume training or not", default=False
    )
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        help="path to the checkpoint",
        default="checkpoints/best.pt",
    )
    parser.add_argument(
        "--save_all", type=bool, help="save all checkpoints or not", default=False
    )
    parser.add_argument(
        "--num_position", type=int, help="number of projection tokens", default=512,
    )
    parser.add_argument(
        "--workers", type=int, help="number of workers", default=16,
    )
    parser.add_argument(
        "--num_epochs", type=int, help="number of epochs", default=120,
    )
    parser.add_argument(
        "--num_classes", type=int, help="number of classes samples", default=23,
    )
    parser.add_argument(
        "--gc", type=int, help="gradient accumulation", default=2,
    )
    args = parser.parse_args()

    wandb.init(
        project="Unseen-Modalities-VL",
        name='Base',
        config={
        "learning_rate": args.lr,
        "epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "gc": args.gc,
        "resume_checkpoint": args.resume_checkpoint,
        "resume_training": args.resume_training,
        "dataset": args.dataset,
        }
    )

    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)
    warnings.filterwarnings("ignore")

    device = "cuda"  # or 'cpu'
    device = torch.device(device)

    base_path = "/work/tesi_asaporita/UnseenModalities-VL/checkpoints/"
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    batch_size = args.batch_size #96

    """
    Multimodal Transfomer
    """
    multimodal_model = ViT(
        num_classes=args.num_classes,
        dim=256,
        depth=6,
        heads=8,
        mlp_dim=512,
        num_position=args.num_position,
    )
    multimodal_model = torch.nn.DataParallel(multimodal_model)
    multimodal_model = multimodal_model.to(device) 

    """
    Feature projection: project unimodal embeddings into a common feature space (K^m, d^m)-dim
    """
    reorganization_module = ViTReorganization(
        dim=256,
        depth=6,
        heads=8,
        mlp_dim=512,
        num_position=args.num_position, #512
    )
    reorganization_module = torch.nn.DataParallel(reorganization_module)
    reorganization_module = reorganization_module.to(device) 

    loss_fn = LabelSmoothLoss(smoothing=0.1) #loss supervised
    loss_fn = loss_fn.cuda()

    optim = torch.optim.SGD(
        list(multimodal_model.parameters())+list(reorganization_module.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4
    )

    scheduler = MultiStepLR(optim, milestones=[70], gamma=0.1)
    scaler = GradScaler()
    BestLoss = float("inf")
    initial_epoch = 0
    BestEpoch = 0
    BestMetric = float("-inf")

    if args.resume_training: 
        print('Restoring checkpoint')
        checkpoint = torch.load(args.resume_checkpoint)
        multimodal_model.load_state_dict(checkpoint["model"])
        reorganization_module.load_state_dict(checkpoint["reorganization"])
        optim.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler']) 
        scheduler.load_state_dict(checkpoint['scheduler'])
        initial_epoch = checkpoint['epoch'] + 1
        BestLoss = checkpoint['best_loss']
        BestMetric = checkpoint['best_metric']
    

    log_path = "logs/{}.csv".format(args.save_name)
    
    print("---------------Start Training---------------")
    with open(log_path, "a") as f:
        for epoch_i in range(initial_epoch, args.num_epochs):
            print("Epoch: %02d" % epoch_i)
            for split in ["train", "val"]:
                acc = 0
                f1_metric = F1_Score()
                auroc = AUROC()
                count = 0
                total_loss = 0
                loss = 0
                print(split)
                multimodal_model.train(split == "train")
                reorganization_module.train(split == "train")  

                if split == 'train':
                    path = args.train_data_path
                    n=args.n_train_samples
                elif split == 'val':
                    path = args.val_data_path 
                    n=args.n_val_samples  
                else:
                    raise NotImplementedError()
    
                ds = wds.DataPipeline(
                    wds.SimpleShardList(braceexpand(path)),
                    wds.tarfile_to_samples(),
                    wds.split_by_worker,
                    wds.split_by_node,
                ).with_length(n)

                if split=='train':
                    if args.dataset=='mmimdb':
                        dataloader = wds.WebLoader(ds, batch_size=batch_size, num_workers=args.workers, pin_memory=True).shuffle(1000).to_tuple("__key__", "image_features.pth", "image_mask.pth", "image_pseudo.pth", "text_features.pth", "text_mask.pth", "text_pseudo.pth", "label.pth")
                    else:
                        dataloader = wds.WebLoader(ds, batch_size=batch_size, num_workers=args.workers, pin_memory=True).shuffle(1000).to_tuple("__key__", "image_features.pth", "image_mask.pth", "image_pseudo.pth", "text_features.pth", "text_mask.pth", "text_pseudo.pth", "label.id")
                else:
                    if args.dataset=='mmimdb':
                        dataloader = wds.WebLoader(ds, batch_size=batch_size, num_workers=args.workers, pin_memory=True).shuffle(1000).to_tuple("__key__", "image_features.pth", "image_mask.pth", "text_features.pth", "text_mask.pth", "label.pth")
                    else:
                        dataloader = wds.WebLoader(ds, batch_size=batch_size, num_workers=args.workers, pin_memory=True).shuffle(1000).to_tuple("__key__", "image_features.pth", "image_mask.pth", "text_features.pth", "text_mask.pth", "label.id")
        
                num_batches =  math.ceil(ds.size/batch_size)

                with tqdm.tqdm(total=num_batches) as pbar:
                    for (i,sample) in enumerate(dataloader):
                        if split=='train':
                            keys, image_features, image_mask, image_pseudo, text_features, text_mask, text_pseudo, label = sample
                        else:
                            keys, image_features, image_mask, text_features, text_mask, label = sample

                        #------------Features------------
                        image_features = [wds.torch_loads(item) for item in image_features]
                        image_features = torch.stack(image_features , dim=0)
                        text_features = [wds.torch_loads(item) for item in text_features]
                        text_features = torch.stack(text_features , dim=0)
                        data = {
                            "Image": image_features,
                            "Text": text_features,
                        }
                        data = dict_to_cuda(data) 
                        #------------Masks------------
                        image_mask = [wds.torch_loads(item) for item in image_mask] 
                        image_mask = torch.stack(image_mask , dim=0)
                        text_mask = [wds.torch_loads(item) for item in text_mask] 
                        text_mask = torch.stack(text_mask , dim=0)
                        masks = {
                            "Image": image_mask,
                            "Text": text_mask,
                        }
                        masks = dict_to_cuda(masks) 

                        #------------Labels------------
                        if args.dataset=='mmimdb':
                            labels = [wds.torch_loads(item) for item in label] #(B, 23)
                            labels = torch.stack(labels , dim=0).cuda()
                        else: 
                            labels = [int(item.decode()) for item in label] #(B, )
                            labels = torch.tensor(labels).cuda()

                        #------------Train Step------------
                        if split == "train":
                            outputs, loss = train_one_step(
                                data,
                                labels,
                                masks,
                                multimodal_model,
                                reorganization_module,
                                optim,
                                loss_fn,
                                scaler,
                                i,
                                num_batches,
                                args.gc,
                                args.dataset,
                            )

                        #------------Validation Step------------
                        else:  #val
                            outputs, loss = val_one_step(
                                data,
                                labels,
                                masks,
                                multimodal_model,
                                reorganization_module,
                                loss_fn,
                                args.gc,
                                args.dataset,
                            )
                        
                        wandb.log({"{}/step_loss".format(split): loss}) #step loss
                        total_loss += loss.item() * batch_size
                        count += outputs.size()[0]
                        
                        if(args.dataset=='mmimdb'):
                            outputs = torch.mean(outputs, dim=1)
                            f1_metric.update(outputs.float(), labels.float()) 
                            F1_Micro, F1_Macro, F1_Samples, F1_Weighted = f1_metric.compute()
                            metric = F1_Macro.item()
                        elif args.dataset == 'Hatefull_Memes':
                            outputs = torch.softmax(outputs, dim=-1)
                            outputs = torch.mean(outputs, dim=1)
                            auroc.update(outputs, labels)
                            aur = auroc.compute()
                            metric = aur.item()
                        elif(args.dataset=='Food101'):
                            outputs = torch.softmax(outputs, dim=-1)    #(B, 2, num_classes)
                            outputs = torch.mean(outputs, dim=1)        #(B, 1, num_classes) = mean of the predictions of the two CLS tokens 
                            _, predict = torch.max(outputs, dim=1)
                            acc1 = (predict == labels).sum().item()
                            acc += int(acc1)
                            metric = acc / float(count)
                        else:
                            raise NotImplementedError

                        pbar.set_postfix_str(
                            "Average loss: {:.4f}, Current loss: {:.4f}, Metric: {:.4f}".format(
                                total_loss / float(count),
                                loss.item(),
                                metric,
                            )
                        )
                        pbar.update()
                    
                    if(args.dataset=='mmimdb'):
                        F1_Micro, F1_Macro, F1_Samples, F1_Weighted = f1_metric.compute()
                        metric = F1_Macro.item()
                    elif(args.dataset=='Food101'):
                        metric = acc / float(count)
                    elif args.dataset == 'Hatefull_Memes':
                        aur = auroc.compute()
                        metric = aur.item()
                    else:
                        raise NotImplementedError
                    
                    f.write(
                        "{},{},{},{}\n".format(
                            epoch_i,
                            split,
                            total_loss / float(count),
                            metric,
                        )
                    )
                    f.flush()
                    wandb.log({"{}/loss".format(split): total_loss / float(count), "{}/loss_epoch".format(split): epoch_i}) #epoch loss
                    wandb.log({"{}/metric".format(split): metric, "{}/metric_epoch".format(split): epoch_i}) #epoch accuracy 

            #scheduler.step()
            wandb.log({"train/lr": scheduler.get_last_lr()[0]}) #epoch lr
            #wandb.log({"train/lr_epoch": epoch_i})

            if metric > BestMetric: 
                BestLoss = total_loss / float(count)
                BestEpoch = epoch_i
                BestMetric = metric
                save = {
                    "epoch": epoch_i,
                    "model": multimodal_model.state_dict(),
                    "reorganization": reorganization_module.state_dict(),
                    "optimizer": optim.state_dict(),
                    "scaler": scaler.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_loss": BestLoss,
                    "best_metric": BestMetric,
                }

                torch.save(
                    save, base_path + "best_multimodal{}{}.pt".format(args.save_name, epoch_i)
                )  
                   
            if args.save_all and epoch_i % 4 == 0: #save model every 4 epochs
                save = {
                    "epoch": epoch_i,
                    "model": multimodal_model.state_dict(),
                    "reorganization": reorganization_module.state_dict(),
                    "optimizer": optim.state_dict(),
                    "scaler": scaler.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_loss": BestLoss,
                    "best_metric": BestMetric,
                }

                torch.save(
                    save, base_path + "best_multimodal{}{}.pt".format(args.save_name, epoch_i)
                )  
    f.close()
