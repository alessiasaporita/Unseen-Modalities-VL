from unimodal.food101_train_dataset import FOOD101DatasetTrain
from unimodal.food101_val_dataset import FOOD101DatasetValidation
from unimodal.mmimdb_train_dataset import MMIMDBDatasetTrain
from unimodal.mmimdb_val_dataset import MMIMDBDatasetVal
from unimodal.hatememes_train_dataset import HateMemesDatasetTrain
from unimodal.hatememes_val_dataset import HateMemesDatasetVal
import torch
import argparse
import tqdm
import os
import shutil
import numpy as np
import torch.nn as nn
import random
import warnings
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
import wandb
import os
from torch.optim.lr_scheduler import MultiStepLR
from utils.heads import MLP
import unimodal.clip as clip
from utils.my_metrics import AUROC
from utils.my_metrics import F1_Score
from transformers import get_polynomial_decay_schedule_with_warmup
"""
def set_schedule(args, optimizer, train_dataloader):
    lr_mult = args.lr_mult #1
    end_lr = args.end_lr #0
    decay_power = 1

    max_steps = len(train_dataloader) * args.num_epochs // args.gc
    warmup_steps = args.warmup_steps
    if isinstance(args.warmup_steps, float):
        warmup_steps = int(max_steps * warmup_steps)

    #It combines a warm-up phase with a polynomial decay schedule for the learning rate.
    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
        lr_end=end_lr,
        power=decay_power,
    )

    return scheduler
"""

def save_pseudo_labels(outputs, keys, modality, dataset):
    detached_outputs = outputs.float().detach().cpu()
    if(dataset != 'mmimdb'):
        detached_outputs = torch.softmax(detached_outputs, dim=-1) 

    #For each sample in the batch, save its relative prediction
    for i in range(len(keys)): 
        #------------RGB------------
        if modality=='image':  
            save_path = "/work/tesi_asaporita/UnseenModalities-VL/{}/image/image_pseudo/{}.npy".format(dataset, keys[i])
            if os.path.exists(save_path): #predictions for i-th sample 
                image_pseudo = np.load(save_path)
                if image_pseudo.shape[0]>=40:
                    image_pseudo=image_pseudo[-39:]
                image_pseudo = np.concatenate((image_pseudo, detached_outputs[i].unsqueeze(0).numpy()))
            else:
                image_pseudo=detached_outputs[i].unsqueeze(0).numpy()
            np.save("/work/tesi_asaporita/UnseenModalities-VL/{}/image/image_pseudo/{}.npy".format(dataset, keys[i]), image_pseudo)
        #------------Audio------------
        else: #Text 
            save_path = "/work/tesi_asaporita/UnseenModalities-VL/{}/text/text_pseudo/{}.npy".format(dataset, keys[i])
            if os.path.exists(save_path):
                text_pseudo = np.load(save_path)
                if text_pseudo.shape[0]>=40:
                    text_pseudo=text_pseudo[-39:]
                text_pseudo = np.concatenate((text_pseudo, detached_outputs[i].unsqueeze(0).numpy()))
            else:
                text_pseudo=detached_outputs[i].unsqueeze(0).numpy()
            np.save("/work/tesi_asaporita/UnseenModalities-VL/{}/text/text_pseudo/{}.npy".format(dataset, keys[i]), text_pseudo)

class LabelSmoothLoss(nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing #0.1
    
    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * self.smoothing / (input.size(-1) - 1.0)
        weight.scatter_(-1, target.unsqueeze(-1), (1.0 - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss

def train_one_step(
    data,
    labels,
    model,
    head,
    optim,
    #scheduler,
    loss_fn,
    scaler,
    indice,
    last_indice,
    gc,
    modality,
):
    with torch.cuda.amp.autocast():
        if modality=='text':
            features = model.encode_text(data) #(B, 77)
        else:
            features = model.encode_image(data) #(B, 512)

        outputs = head(features) #(B, num_classes)
        labels = labels.float()
        #Total loss
        output_loss = loss = F.binary_cross_entropy_with_logits(outputs, labels) #loss_fn(outputs, labels) 
    
    loss = loss / gc
    #loss.backward()
    scaler.scale(loss).backward()

    if((indice + 1) % gc == 0) or (indice + 1 == last_indice):
        #optim.step()
        scaler.step(optim)
        scaler.update()
        optim.zero_grad()
        #scheduler.step()            
        #wandb.log({"train/lr": scheduler.get_last_lr()[0]})

    return outputs, output_loss


def val_one_step(
    data,
    labels,
    model,
    head,
    loss_fn,
    modality,
):
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            if modality=='text':
                features = model.encode_text(data) 
            else:
                features = model.encode_image(data)
            #features = features / features.norm(dim=1, keepdim=True)
                
            outputs = head(features)
            labels = labels.float()
            output_loss = loss = F.binary_cross_entropy_with_logits(outputs, labels) #loss_fn(outputs, labels) 
    
    return outputs, output_loss



if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lr", type=float, help="learning rate", default=5e-6
    )  
    parser.add_argument("--batch_size", type=int, help="batch size", default=1024)
    parser.add_argument("--data-root", type=str, default='/work/tesi_asaporita/MissingModalities/datasets') 
    parser.add_argument(
        "--save_name", type=str, help="name to save the model", default="1e-1",
    ) 
    parser.add_argument(
        "--modality", type=str, help="text or image", default="image",
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
        "--num_classes", type=int, help="number of classes samples", default=23,
    )
    parser.add_argument("--gc", default=1, type=int)
    parser.add_argument("--num_workers", default=16, type=int)
    parser.add_argument("--num_epochs", default=20, type=int) 
    parser.add_argument("--dataset", default='mmimdb', type=str, choices=["mmimdb", "Hatefull_Memes", "Food101"])
    parser.add_argument("--clip_model", default='ViT-B/32', type=str, choices=['RN50x16', 'ViT-B/32'])
    parser.add_argument("--dim", default=512, type=int)

    parser.add_argument("--image_size", default=384, type=int)
    parser.add_argument("--warmup-steps", default=0.1, type=float)
    parser.add_argument("--max-step", default=None, type=int)
    parser.add_argument("--end-lr", default=0, type=float)
    parser.add_argument("--lr-mult", default=1, type=float)
    

    args = parser.parse_args()

    wandb.init(
        project="Unseen-Modalities-VL",
        name='Unimodal',
        config={
        "modality":args.modality,
        "learning_rate": args.lr,
        "epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "gc": args.gc,
        "resume_checkpoint": args.resume_checkpoint,
        "resume_training": args.resume_training,
        }
    )


    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)
    warnings.filterwarnings("ignore")

    device = "cuda"  # or 'cpu'
    device = torch.device(device)

    base_path = "/work/tesi_asaporita/UnseenModalities-VL/{}/{}/checkpoint/".format(args.dataset, args.modality)
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    batch_size = args.batch_size #1024

    model, preprocess = clip.load(args.clip_model, device=device)
    model = model.float()
    
    if args.dataset=='mmimdb':
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = LabelSmoothLoss(smoothing=0.1) #loss supervised
    loss_fn = loss_fn.cuda()

    head = MLP(args.dim, args.num_classes).to(device)
    
    if args.modality == 'image':
        for name, param in model.named_parameters():
            if "visual" not in name:
                param.requires_grad = False
    else: #text modality
        for param in model.visual.parameters():
            param.requires_grad = False

    params_to_train = [param for param in model.parameters() if param.requires_grad]
    for param in head.parameters(): 
        params_to_train.append(param)
    optim = torch.optim.AdamW(params_to_train, lr=args.lr, betas=(0.9, 0.98), eps=1e-6,
                                weight_decay=0.2)
        
    scaler = GradScaler()
    BestLoss = float("inf")
    initial_epoch = 0
    BestEpoch = 0
    BestMetric = 0

    if args.resume_training: 
        print('Restoring checkpoint')
        checkpoint = torch.load(args.resume_checkpoint)
        head.load_state_dict(checkpoint['head'])
        optim.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler']) ######
        initial_epoch = checkpoint['epoch'] + 1
        BestLoss = checkpoint['best_loss']
        BestMetric = checkpoint['best_metric']
        if args.modality=='image':
            model.visual.load_state_dict(checkpoint["visual"])
        else: #text modality
            for name, param in model.named_parameters():
                if name in checkpoint:
                    param.data.copy_(checkpoint[name].data)

    if args.dataset=='Food101':  
        dataset_train = FOOD101DatasetTrain(
            data_dir=args.data_root,
            image_size=args.image_size,
            split="train",
            modality=args.modality,
            preprocess=preprocess, 
        )
        dataset_val = FOOD101DatasetValidation(
            data_dir=args.data_root,
            image_size=args.image_size,
            split="val",
            preprocess=preprocess,
        )
    elif args.dataset=='mmimdb':
        dataset_train = MMIMDBDatasetTrain(
            data_dir=args.data_root,
            image_size=args.image_size,
            split="train",
            modality=args.modality,
            preprocess=preprocess, 
        )
        dataset_val = MMIMDBDatasetVal(
            data_dir=args.data_root,
            image_size=args.image_size,
            modality=args.modality,
            split="val",
            preprocess=preprocess,
        )
    elif args.dataset=='Hatefull_Memes':
        dataset_train = HateMemesDatasetTrain(
            data_dir=args.data_root,
            image_size=args.image_size,
            split="train",
            modality=args.modality,
            preprocess=preprocess, 
        )
        dataset_val = HateMemesDatasetVal(
            data_dir=args.data_root,
            image_size=args.image_size,
            split="val",
            preprocess=preprocess,
        )
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    dataloaders = {"train": train_loader, "val": val_loader}
    log_path = "logs/{}.csv".format(args.save_name)
    #scheduler = set_schedule(args, optimizer=optim, train_dataloader=train_loader)

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
                model.train(split == "train")
                head.train(split == "train")

                with tqdm.tqdm(total=len(dataloaders[split])) as pbar:
                    for i, sample in enumerate(dataloaders[split]):
                        labels = sample['label'].cuda() #(B, )
                        keys = sample['key']

                        if args.modality=='text':
                            text = sample['text']
                            data = clip.tokenize(
                                text, truncate=True).to(device, non_blocking=True) #(B, 77)
                        else:
                            data = sample['image'][0].cuda()

                        if split == 'train':
                            output, loss = train_one_step(
                                data = data,
                                labels=labels,
                                model=model,
                                head=head,
                                optim=optim,
                                #scheduler=scheduler,
                                loss_fn=loss_fn,
                                scaler=scaler,
                                indice=i,
                                last_indice=len(dataloaders[split]),
                                gc=args.gc,
                                modality=args.modality,
                            )
                            save_pseudo_labels(output, keys, args.modality, args.dataset)
                        else:
                            output, loss = val_one_step(
                                data=data,
                                labels=labels,
                                model=model,
                                head=head,
                                loss_fn=loss_fn,
                                modality=args.modality
                            )
                            
                        wandb.log({"{}/step_loss".format(split): loss}) #step loss
                        total_loss += loss.item() * batch_size
                        count += output.size()[0]


                        if(args.dataset=='Food101'):
                            outputs = torch.softmax(output.float(), dim=-1) #(B, n_classes)
                            _, predict = torch.max(outputs, dim=1)
                            acc1 = (predict == labels).sum().item()
                            acc += int(acc1)
                            metric = acc / float(count)
                        elif args.dataset == 'Hatefull_Memes':
                            auroc.update(output.float(), labels)
                            aur = auroc.compute(use_softmax=True)
                            metric = aur.item()
                        elif(args.dataset=='mmimdb'):
                            f1_metric.update(output.float(), labels.float()) 
                            F1_Micro, F1_Macro, F1_Samples, F1_Weighted = f1_metric.compute()
                            metric = F1_Macro.item()
                        
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
                        aur = auroc.compute(use_softmax=True)
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
                    wandb.log({"{}/metric".format(split): metric, "{}/metric_epoch".format(split): epoch_i}) #epoch metric 

            if metric > BestMetric: 
                BestLoss = total_loss / float(count)
                BestEpoch = epoch_i
                BestMetric = metric
                if args.modality=='image':
                    save = {
                        "visual": model.visual.state_dict(),
                        "head": head.state_dict(),
                        "epoch": epoch_i,
                        "optimizer": optim.state_dict(),
                        "scaler": scaler.state_dict(),
                        "best_loss": BestLoss,
                        "best_metric": BestMetric,
                    }
                else:
                    save = {}
                    for name, param in model.named_parameters():
                        if "visual" not in name:
                            save[name] = param
                    save["epoch"] = epoch_i
                    save["optimizer"] = optim.state_dict()
                    save["scaler"] = scaler.state_dict()
                    save["best_loss"] = BestLoss
                    save["best_metric"] = BestMetric
                    save["head"] = head.state_dict()
                    
                torch.save(
                    save, base_path + "best_unimodal_{}_{}.pt".format(args.save_name, epoch_i)
                )
            if args.save_all and epoch_i % 4 == 0: #save model every 4 epochs
                if args.modality=='image':
                    save = {
                        "visual": model.visual.state_dict(),
                        "head": head.state_dict(),
                        "epoch": epoch_i,
                        "optimizer": optim.state_dict(),
                        "scaler": scaler.state_dict(),
                        "best_loss": BestLoss,
                        "best_metric": BestMetric,
                    }
                else:
                    save = {}
                    for name, param in model.named_parameters():
                        if "visual" not in name:
                            save[name] = param
                    save["epoch"] = epoch_i
                    save["optimizer"] = optim.state_dict()
                    save["scaler"] = scaler.state_dict()
                    save["best_loss"] = BestLoss
                    save["best_metric"] = BestMetric
                    save["head"] = head.state_dict()
                    
                torch.save(
                    save, base_path + "best_unimodal_{}_{}.pt".format(args.save_name, epoch_i)
                )
    f.close()
