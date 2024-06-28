import torch
from torch import nn
import pdb
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViTReorganization(nn.Module): #dim=256, depth=6, heads=8, mlp_dim=512, num_position=512
    def __init__(self, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., num_position = 512):
        super().__init__()

        self.image_embedding = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, dim),
            nn.LayerNorm(dim)
        )
        self.text_embedding = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, dim),
            nn.LayerNorm(dim)
        )

        self.image_pos_embedding = nn.Parameter(torch.randn(1, 50, dim)) #(1, 50, 256)
        self.text_pos_embedding = nn.Parameter(torch.randn(1, 77, dim)) #(1, 77, 256)

        self.pos_embedding = {
            'Audio': self.image_pos_embedding,
            'RGB': self.text_pos_embedding,
        }

        self.dropout = nn.Dropout(emb_dropout)
        
        #dim=256, depth=6, heads=8, mlp_dim=512, dropout=0
        self.image_transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)  
        self.text_transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.transformer = {
            'Image': self.image_transformer,
            'Text': self.text_transformer,
        }

        self.pool = pool
        self.to_latent = nn.Identity()

        self.image_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_position)
        )

        self.text_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_position)
        )

        self.heads = {
            'Image': self.image_head,
            'Text': self.text_head,
        }
        #fc_layer implemented by us
        self.image_proj = nn.Sequential(
            nn.LayerNorm(512), 
            nn.Linear(512, dim),
            nn.LayerNorm(dim)
        )
        self.text_proj = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, dim),
            nn.LayerNorm(dim)
        ) 

    #reorganization --> (B, 512, 50) * (B, 50, 512) -> (B, 512, 512)
    def reorganization(self, feature, position):
        feature = torch.matmul(         
            feature.transpose(1, 2).contiguous(), position
        ) 
        feature = feature.transpose(1, 2).contiguous() #(B, 512, 512) = (k*, d)
        return feature

    def forward(self, image_inputs, text_inputs): # (B, 50, 512) / (B, 77, 512)

        image = self.image_embedding(image_inputs) + self.image_pos_embedding                   #(B, 50, 256) 
        text = self.text_embedding(text_inputs) + self.text_pos_embedding                       #(B, 77, 256)

        image = self.dropout(image)
        text = self.dropout(text)

        image = self.image_transformer(image)             #(B, 50, 256)
        text = self.text_transformer(text)                #(B, 77, 256)

        image = self.image_head(image)        #(B, 50, 512) = (k x k*)
        text = self.text_head(text)           #(B, 77, 512) = (k x k*)

        image = torch.softmax(image, dim = -1)
        text = torch.softmax(text, dim=-1)

        image = self.reorganization(image_inputs, image)          #(B, 512, 512)
        text = self.reorganization(text_inputs, text)             #(B, 512, 512) 

        #this part of code was missing from the researchers and was implemented by us 
        image = self.image_proj(image)        #(B, 512, 256) = (k*, d*)
        text = self.text_proj(text)           #(B, 512, 256) = (k*, d*)
  
        return image, text