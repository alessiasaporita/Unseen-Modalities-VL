import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.classifier = nn.Sequential(
                nn.Linear(input_size, input_size * 2),
                nn.LayerNorm(input_size * 2),
                nn.GELU(),
                nn.Linear(input_size * 2, output_size),
            )
        self.classifier.apply(init_weights)  

    def forward(self, x):
        return self.classifier(x)
    
class MLP_mod(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, output_size)
        )
        self.classifier.apply(init_weights)  
    
    def forward(self, x):
        return self.classifier(x)

class FCModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(FCModel, self).__init__()
        self.classifier = nn.Linear(input_size, output_size)
        #self.classifier.apply(init_weights)  

    def forward(self, x):
        return self.classifier(x)
    


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

def get_classifier(args, device):
    #CLASSIFIER
    if(args.dataset=='mmimdb'):
        if args.model == 'FCModel':
            classifier = FCModel(args.input_size, args.mmimdb_class_num).to(device)
        else:
            classifier = MLP(args.input_size, args.mmimdb_class_num).to(device)

    elif(args.dataset=='Food101'):
        if args.model == 'FCModel':
            classifier = FCModel(args.input_size, args.food101_class_num).to(device)
        else:
            classifier = MLP(args.input_size, args.food101_class_num).to(device)

    else: #Hatefull_memes
        if args.model == 'FCModel':
            classifier = FCModel(args.input_size, args.hatememes_class_num).to(device)
        else:
            classifier = MLP(args.input_size, args.hatememes_class_num).to(device)
    
    return classifier