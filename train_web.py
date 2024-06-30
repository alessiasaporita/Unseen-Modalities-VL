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
from utils.my_metrics import F1_Score, AUROC
import torch.nn.functional as F

"""
    https://github.com/gerasmark/Reproducing-Unseen-Modality-Interaction/blob/main/main.ipynb
"""

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

class AlignmentModule(nn.Module):
    def __init__(self, dim=256, num_classes=101):
        super(AlignmentModule, self).__init__()
        self.base_vectors = nn.Parameter(torch.randn(1, num_classes, 256)) #(1, num_classes, d*), class tokens to be learnt 

    def forward(self, input): # input [B, 512, 256]
        input = torch.mean(input, dim=1, keepdim=True) # [B, 1, d*], mean vectors of the samples in the batch
        base_vectors = self.base_vectors.repeat(input.size()[0], 1, 1) #[B, num_classes, d*], class tokens
        sim = torch.mean((base_vectors - input) ** 2, dim=-1) #[B, num_classes], euclidean distance between the average vectors of the batch and each class tokens
        return sim 

def train_one_step(
    data,
    labels,
    masks,
    image_pseudo,
    text_pseudo,
    multimodal_model,
    reorganization_module,
    alignment_model,
    optim,
    loss_fn,
    kl_loss_fn,
    scaler,
    indice,
    last_indice,
    gc,
    deactivate_KL,
    num_classes,
    dataset,
    base,
):
    image, text = reorganization_module( #feature projection = (B, 512, 256) = (B, k*, d*)
        data['Image'], data['Text'] 
    ) 

    image_sim = alignment_model(image) #sim = [B, num_classes]
    text_sim = alignment_model(text) #sim = [B, num_classes]

    outputs = multimodal_model( #(B, 2, num_classes), the two CLS tokens
        image, text, masks['Image'], masks['Text']
    ) 

    #ALIGNMENT LOSS: max similarity between average vectors of the samples and the corrisponding class tokens, ie min euclidean distance between average vectors and the class tokens
    image_indices = torch.sum(masks['Image'].squeeze(-1), dim=-1) > 0 #(B,) 
    text_indices = torch.sum(masks['Text'].squeeze(-1), dim=-1) > 0 #(B,) 

    image_labels = labels[image_indices]    #(number of image samples, )
    text_labels = labels[text_indices]        #(number of text samples, ) 
    
    #Audio and RGB distances
    image_sim = image_sim[image_indices]            #(number of image samples, num_classes) 
    text_sim = text_sim[text_indices]                  #(number of text samples, num_classes)
    
    if dataset=='mmimdb':
        image_sim = torch.sum(image_sim * image_labels, dim=-1)      #(number of image samples, ) 
        text_sim = torch.sum(text_sim * text_labels, dim=-1)            #(number of text samples, )
        alignment_loss = (torch.sum(image_sim) + torch.sum(text_sim)) / (torch.sum(image_labels) + torch.sum(text_labels))
    else:
        image_onehot_labels = F.one_hot(image_labels, num_classes = num_classes)       #(number of image samples, num_classes) 
        text_onehot_labels = F.one_hot(text_labels, num_classes = num_classes)           #(number of text samples, num_classes)
        image_sim = torch.sum(image_sim * image_onehot_labels, dim=-1)      #(number of image samples, ) 
        text_sim = torch.sum(text_sim * text_onehot_labels, dim=-1)            #(number of text samples, )
        alignment_loss = (torch.sum(image_sim) + torch.sum(text_sim)) / (torch.sum(image_indices) + torch.sum(text_indices))

    #Total Loss: L-supervised + gamma L-pseudo + alpha L-align, with gamma = 3000, alpha = 0.001 
    if base: #Supervised Loss
        if dataset=='mmimdb':
            labels=labels.float()
            output_loss = loss = (F.binary_cross_entropy_with_logits(outputs[:,0], labels) + F.binary_cross_entropy_with_logits(outputs[:,1], labels)) * 0.5 
        else:
            output_loss = loss = (loss_fn(outputs[:,0], labels) + loss_fn(outputs[:,1], labels)) * 0.5 
    elif deactivate_KL: #+ Align Loss
        if dataset =='mmimdb':
            labels=labels.float()
            output_loss = loss = (F.binary_cross_entropy_with_logits(outputs[:,0], labels) + F.binary_cross_entropy_with_logits(outputs[:,1], labels)) * 0.5 +  0.001 * alignment_loss
        else:
            output_loss = loss = (loss_fn(outputs[:,0], labels) + loss_fn(outputs[:,1], labels)) * 0.5 +  0.001 * alignment_loss
    else: #+ Pseudo Loss
        #L_pseudo: mean KL-divergence between log-prob of audio and pseudo label, rgb and rgb pseudo label, multiplied for their weigths=number of rgb/audio samples
        image_pseudo = image_pseudo[image_indices]
        text_pseudo = text_pseudo[text_indices]
        if dataset =='mmimdb':
            prob = outputs[:,1]
            image_prob = probs[image_indices]               #(number of image samples, num_classes) 
            text_prob = probs[text_indices]                   #(number of text samples, num_classes) 
            BCE=0
            if torch.sum(image_prob) != 0:
                BCE += torch.mean(F.binary_cross_entropy_with_logits(image_prob, image_pseudo)) * torch.sum(image_indices) 
            if torch.sum(text_indices) != 0:
                BCE += torch.mean(F.binary_cross_entropy_with_logits(text_prob, text_pseudo)) * torch.sum(text_indices)
            labels=labels.float()
            output_loss = loss = F.binary_cross_entropy_with_logits(outputs[:,0], labels) + BCE / labels.size()[0] * 0.001 +  0.001 * alignment_loss
        else:
            probs = torch.softmax(outputs[:,1], dim=-1)
            image_prob = probs[image_indices]               #(number of image samples, num_classes) 
            text_prob = probs[text_indices]                   #(number of text samples, num_classes) 
            kl_loss=0
            if torch.sum(image_prob) != 0:
                kl_loss += torch.mean(kl_loss_fn(torch.log(image_prob), image_pseudo)) * torch.sum(image_indices) #Food101: 0.0812/ HM: 0.6607
            if torch.sum(text_indices) != 0:
                kl_loss += torch.mean(kl_loss_fn(torch.log(text_prob), text_pseudo)) * torch.sum(text_indices) #Food101: 0.2267/ HM: 0.9412
            
            #Total loss
            output_loss = loss = loss_fn(outputs[:,0], labels) + kl_loss / labels.size()[0] * 50 +  0.001 * alignment_loss
    
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
    alignment_model,
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
    parser.add_argument("--deactivate_KL", type=bool, help="Deactivate KL loss", default=False)
    parser.add_argument("--base", type=bool, help="Deactivate KL and align loss", default=False)
    parser.add_argument(
        "--train_data_path",
        type=str,
        help="path to train data",
        default="/work/tesi_asaporita/webdataset/food101-training-{000..020}.tar",
    )
    parser.add_argument(
        "--val_data_path",
        type=str,
        help="path to validation data",
        default="/work/tesi_asaporita/webdataset/food101-validation-{000..004}.tar",
    )
    parser.add_argument(
        "--n_train_samples", type=int, help="number of training samples", choices=[61127, 32278, 8500], default=61127,
    )
    parser.add_argument(
        "--n_val_samples", type=int, help="number of training samples", choices=[6845, 5411, 500], default=6845,
    )
    parser.add_argument(
        "--num_classes", type=int, help="number of classes samples", default=101,
    )
    parser.add_argument("--dataset", default='Food101', type=str, choices=["mmimdb", "Hatefull_Memes", "Food101"])
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
        "--num_epochs", type=int, help="number of epochs", default=50,
    )
    parser.add_argument(
        "--gc", type=int, help="gradient accumulation", default=2,
    )
    args = parser.parse_args()

    wandb.init(
        project="Unseen-Modalities-VL",
        name='VL',
        config={
        "learning_rate": args.lr,
        "epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "gc": args.gc,
        "resume_checkpoint": args.resume_checkpoint,
        "resume_training": args.resume_training,
        "deactivate_KL": args.deactivate_KL,
        "dataset": args.dataset, 
        }
    )

    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)
    warnings.filterwarnings("ignore")

    device = "cuda"  # or 'cpu'
    device = torch.device(device)

    base_path = "/work/tesi_asaporita/checkpoint/{}/".format(args.dataset)
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
        num_position=args.num_position,
    )
    reorganization_module = torch.nn.DataParallel(reorganization_module) 
    reorganization_module = reorganization_module.to(device) 
    

    """
    Alignment: align embeddings with learnable class tokens 
    """
    alignment_model = AlignmentModule(num_classes = args.num_classes) 
    alignment_model = torch.nn.DataParallel(alignment_model)
    alignment_model = alignment_model.to(device) 


    loss_fn = LabelSmoothLoss(smoothing=0.1) #loss supervised
    loss_fn = loss_fn.cuda()

    kl_loss_fn = nn.KLDivLoss(reduce=False) #loss pseudolabel
    kl_loss_fn = kl_loss_fn.cuda()

    optim = torch.optim.SGD(
        list(multimodal_model.parameters())+list(reorganization_module.parameters()) + list(alignment_model.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = MultiStepLR(optim, milestones=[70], gamma=0.1)
    scaler = GradScaler()
    BestLoss = float("inf")
    initial_epoch = 0
    BestEpoch = 0
    BestMetric = 0

    if args.resume_training: 
        print('Restoring checkpoint')
        checkpoint = torch.load(args.resume_checkpoint)
        multimodal_model.load_state_dict(checkpoint["model"])
        reorganization_module.load_state_dict(checkpoint["reorganization"])
        alignment_model.load_state_dict(checkpoint["alignment"])
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
                alignment_model.train(split == "train") 

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
                        data = dict_to_cuda(data) #dict with Image =(B, 50, 512), Text=(B, 77, 512)

                        #------------Masks------------
                        image_mask = [wds.torch_loads(item) for item in image_mask] 
                        image_mask = torch.stack(image_mask , dim=0)
                        text_mask = [wds.torch_loads(item) for item in text_mask] 
                        text_mask = torch.stack(text_mask , dim=0)
                        masks = {
                            "Image": image_mask,
                            "Text": text_mask,
                        }
                        masks = dict_to_cuda(masks) #dict with 'RGB'=(B, 512, 1), Audio=(B, 512, 1)
                        
                        #------------Labels------------
                        if args.dataset=='mmimdb':
                            labels = [wds.torch_loads(item) for item in label] #(B, 23)
                            labels = torch.stack(labels , dim=0).cuda()
                        else: 
                            labels = [int(item.decode()) for item in label] #(B, )
                            labels = torch.tensor(labels).cuda()

                        #------------Train Step------------
                        if split == "train":
                            #------------Pseudo Labels------------
                            #keys = [s.split('__')[1] for s in keys]
                            #audio_pseudo, rgb_pseudo = get_pseudo_labels(keys, masks, args.deactivate_KL)
                            
                            if not args.deactivate_KL:
                                image_pseudo = [wds.torch_loads(item) for item in image_pseudo] 
                                image_pseudo = torch.stack(image_pseudo , dim=0)
                                text_pseudo = [wds.torch_loads(item) for item in text_pseudo] 
                                text_pseudo = torch.stack(text_pseudo , dim=0)

                                image_pseudo = image_pseudo.cuda() #(num_classes, )
                                text_pseudo = text_pseudo.cuda() #(num_classes, )

                            outputs, loss = train_one_step(
                                data,
                                labels,
                                masks,
                                image_pseudo,
                                text_pseudo,
                                multimodal_model,
                                reorganization_module,
                                alignment_model,
                                optim,
                                loss_fn,
                                kl_loss_fn,
                                scaler,
                                i,
                                num_batches,
                                args.gc,
                                args.deactivate_KL,
                                args.num_classes,
                                args.dataset,
                                args.base,
                            )

                        #------------Validation Step------------
                        else:  #val
                            outputs, loss = val_one_step(
                                data,
                                labels,
                                masks,
                                multimodal_model,
                                reorganization_module,
                                alignment_model,
                                loss_fn,
                                args.gc,
                                args.dataset,
                            )
                        
                        wandb.log({"{}/step_loss".format(split): loss})
                        count += outputs.size()[0]
                        total_loss += loss.item() * batch_size

                        if(args.dataset=='mmimdb'):
                            outputs = torch.mean(outputs, dim=1)
                            f1_metric.update(outputs, labels) 
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
            wandb.log({"train/lr": scheduler.get_last_lr()[0]}) 

            if metric > BestMetric: 
                BestLoss = total_loss / float(count)
                BestEpoch = epoch_i
                BestMetric = metric
                save = {
                    "epoch": epoch_i,
                    "model": multimodal_model.state_dict(),
                    "reorganization": reorganization_module.state_dict(),
                    "alignment": alignment_model.state_dict(),
                    "optimizer": optim.state_dict(),
                    "scaler": scaler.state_dict(), ####
                    "scheduler": scheduler.state_dict(), ####
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
                    "alignment": alignment_model.state_dict(),
                    "optimizer": optim.state_dict(),
                    "scaler": scaler.state_dict(), ####
                    "scheduler": scheduler.state_dict(), ####
                    "best_loss": BestLoss,
                    "best_metric": BestMetric,
                }

                torch.save(
                    save, base_path + "best_multimodal{}{}.pt".format(args.save_name, epoch_i)
                )  
    f.close()
