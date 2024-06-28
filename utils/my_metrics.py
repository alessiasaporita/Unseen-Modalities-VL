import torch
from torchmetrics.functional import f1_score, auroc
import numpy as np
#from sklearn.metrics import f1_score

def precision_recall_f1(y_true, y_pred):
    eps = 1e-10
    # True Positives (TP)
    TP = torch.sum((y_true == 1) & (y_pred == 1), axis=0)
    # False Positives (FP)
    FP = torch.sum((y_true == 0) & (y_pred == 1), axis=0)
    # False Negatives (FN)
    FN = torch.sum((y_true == 1) & (y_pred == 0), axis=0)
    
    precision = TP / (TP + FP + eps)  
    recall = TP / (TP + FN + eps)  
    
    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall + eps)  
    
    return precision, recall, f1

class Accuracy():
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits, target):
        logits, target = (
            logits.detach().to(self.correct.device),
            target.detach().to(self.correct.device),
        )
        if logits.size(-1)>1:
            preds = logits.argmax(dim=-1)
        else:
            preds = (torch.sigmoid(logits)>0.5).long()
            
        preds = preds[target != -100]
        target = target[target != -100]
        if target.numel() == 0:
            return 1

        assert preds.shape == target.shape

        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        return self.correct / self.total

class AUROC():
    def __init__(self, dist_sync_on_step=False):
        #super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.correct =torch.tensor(0.0)
        self.logits=[]
        self.targets=[]

    def update(self, logits, target):
        logits, targets = (
            logits.detach().cpu(),
            target.detach().cpu(),
        )
        
        self.logits.append(logits)
        self.targets.append(targets)


    def compute(self, use_softmax=False):
        if type(self.logits) == list:
            all_logits = torch.cat(self.logits)
            all_targets = torch.cat(self.targets).long()
        else:
            all_logits = self.logits
            all_targets = self.targets.long()
    
        if all_logits.size(-1)>1:
            if use_softmax:
                all_logits = torch.softmax(all_logits, dim=1)
            AUROC = auroc(all_logits, all_targets, num_classes=2)
        else:
            all_logits = torch.sigmoid(all_logits)
            AUROC = auroc(all_logits, all_targets)
        
        return AUROC
    
class F1_Score():
    def __init__(self, dist_sync_on_step=False):
        #super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.correct =torch.tensor(0.0)
        self.logits=[]
        self.targets=[]

    def update(self, logits, target):
        logits, targets = (
            logits.detach().cpu(),
            target.detach().cpu(),
        )
        
        self.logits.append(logits)
        self.targets.append(targets)


    def compute(self, use_sigmoid=True):
        if type(self.logits) == list: #True
            all_logits = torch.cat(self.logits)
            all_targets = torch.cat(self.targets).long()
        else:
            all_logits = self.logits
            all_targets = self.targets.long()
        if use_sigmoid:
            all_logits = torch.sigmoid(all_logits)

        F1_Micro = f1_score(all_logits, all_targets, average='micro')
        F1_Macro = f1_score(all_logits, all_targets, average='macro', num_classes=23)
        F1_Samples = f1_score(all_logits, all_targets, average='samples')
        F1_Weighted = f1_score(all_logits, all_targets, average='weighted', num_classes=23)
        return (F1_Micro, F1_Macro, F1_Samples, F1_Weighted)

class My_F1_Macro():
    def __init__(self, threshold=0.5):
        self.logits=[]
        self.targets=[]
        self.threshold=threshold

    def update(self, logits, target):
        logits, targets = (
            logits.detach().cpu(),
            target.detach().cpu(),
        )
        
        self.logits.append(logits)
        self.targets.append(targets)

    def compute(self, use_sigmoid=True):
        all_logits = torch.cat(self.logits)
        all_targets = torch.cat(self.targets).long()
        if use_sigmoid:
            all_logits = torch.sigmoid(all_logits)
        y_pred = (all_logits > self.threshold).type(torch.int64)
        # Calculate precision, recall, and F1 score for each class
        precision, recall, f1_scores = precision_recall_f1(all_targets, y_pred)
        
        # Calculate macro F1 score
        f1_macro = torch.mean(f1_scores)
        
        return f1_macro

   

    
