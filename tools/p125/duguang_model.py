"""读光 ConvNextViT — exact key match to the pretrained checkpoint."""
import torch
from torch import nn
from torch.nn import functional as F


class CLayerNorm(nn.Module):
    """Channels-first LayerNorm with flat weight/bias keys."""
    def __init__(self,dim,eps=1e-6):
        super().__init__()
        self.weight=nn.Parameter(torch.ones(dim))
        self.bias=nn.Parameter(torch.zeros(dim))
        self.eps=eps
    def forward(self,x):
        u=x.mean(1,keepdim=True)
        s=(x-u).pow(2).mean(1,keepdim=True)
        x=(x-u)/torch.sqrt(s+self.eps)
        return x*self.weight[:,None,None]+self.bias[:,None,None]


class ConvNeXtBlock(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.dwconv=nn.Conv2d(dim,dim,7,padding=3,groups=dim)
        self.norm=nn.LayerNorm(dim,eps=1e-6)
        self.pwconv1=nn.Linear(dim,4*dim)
        self.pwconv2=nn.Linear(4*dim,dim)
        self.gamma=nn.Parameter(torch.ones(dim))
    def forward(self,x):
        res=x
        x=self.dwconv(x)
        x=x.permute(0,2,3,1)
        x=self.norm(x)
        x=self.pwconv1(x)
        x=F.gelu(x)
        x=self.pwconv2(x)
        x=self.gamma*x
        x=x.permute(0,3,1,2)
        return res+x


class ConvNeXtCNN(nn.Module):
    def __init__(self,dims=(96,192,256,512),blocks=(3,3,8,3)):
        super().__init__()
        self.downsample_layers=nn.ModuleList()
        stem=nn.Sequential(nn.Conv2d(1,dims[0],kernel_size=4,stride=4),CLayerNorm(dims[0]))
        self.downsample_layers.append(stem)
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                CLayerNorm(dims[i]),nn.Conv2d(dims[i],dims[i+1],kernel_size=(2,1),stride=(2,1))))
        self.stages=nn.ModuleList()
        for i in range(4):
            self.stages.append(nn.ModuleList([ConvNeXtBlock(dims[i]) for _ in range(blocks[i])]))
        self.norm=CLayerNorm(dims[-1])
    def forward(self,x):
        for i in range(4):
            x=self.downsample_layers[i](x)
            for blk in self.stages[i]:x=blk(x)
        return x


class Attention(nn.Module):
    """Multi-head attention with qkv/proj naming to match checkpoint."""
    def __init__(self,dim,heads=8):
        super().__init__()
        self.heads=heads;self.dim=dim
        self.qkv=nn.Linear(dim,dim*3,bias=True)
        self.proj=nn.Linear(dim,dim)
    def forward(self,x):
        B,N,C=x.shape
        qkv=self.qkv(x).reshape(B,N,3,self.heads,C//self.heads).permute(2,0,3,1,4)
        q,k,v=qkv.unbind(0)
        attn=(q@k.transpose(-2,-1))*(C//self.heads)**-0.5
        attn=attn.softmax(dim=-1)
        out=(attn@v).transpose(1,2).reshape(B,N,C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self,dim=192,heads=8):
        super().__init__()
        self.norm1=nn.LayerNorm(dim)
        self.attn=Attention(dim,heads)
        self.norm2=nn.LayerNorm(dim)
        self.mlp=Mlp(dim)
    def forward(self,x):
        x=x+self.attn(self.norm1(x))
        x=x+self.mlp(self.norm2(x))
        return x


class Mlp(nn.Module):
    def __init__(self,dim,hidden=None):
        super().__init__()
        hidden=hidden or dim*4
        self.fc1=nn.Linear(dim,hidden)
        self.fc2=nn.Linear(hidden,dim)
    def forward(self,x):
        return self.fc2(F.gelu(self.fc1(x)))


class ViTSTR(nn.Module):
    def __init__(self,dim=192,depth=12,max_pos=75,vocab_size=7644):
        super().__init__()
        self.patch_embed=nn.Module()
        self.patch_embed.proj=nn.Conv2d(512,dim,1,1)
        self.cls_token=nn.Parameter(torch.randn(1,1,dim)*0.02)
        self.pos_embed=nn.Parameter(torch.randn(1,max_pos,dim)*0.02)
        self.blocks=nn.ModuleList([TransformerBlock(dim) for _ in range(depth)])
        self.norm=nn.LayerNorm(dim)
        self.head=nn.Linear(dim,vocab_size)
    def forward(self,feat):
        feat=self.patch_embed.proj(feat)
        B,C,H,W=feat.shape
        feat=feat.flatten(2).transpose(1,2)
        cls=self.cls_token.expand(B,-1,-1)
        feat=torch.cat([cls,feat],dim=1)
        feat=feat[:,:self.pos_embed.shape[1]]+self.pos_embed[:,:feat.shape[1]]
        for blk in self.blocks:feat=blk(feat)
        feat=self.norm(feat)
        return self.head(feat)


class DuguangOCR(nn.Module):
    def __init__(self,vocab_size=7644):
        super().__init__()
        self.cnn_model=ConvNeXtCNN()
        self.vitstr=ViTSTR(vocab_size=vocab_size)
    def forward(self,x):
        feat=self.cnn_model(x)
        return self.vitstr(feat)
