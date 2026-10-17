"""Plain convolutional U-Net components; independent of src.model.DiT."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def norm(channels):
    return nn.GroupNorm(math.gcd(channels, 32), channels, eps=1e-6)


def timestep_embedding(t, dim=128):
    freq=torch.exp(-math.log(10000)*torch.arange(dim//2,device=t.device)/ (dim//2))
    phase=t.float()[:,None]*freq[None]
    return torch.cat([phase.cos(),phase.sin()],dim=-1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, cond_dim=512):
        super().__init__()
        self.norm1=norm(cin);self.conv1=nn.Conv2d(cin,cout,3,padding=1)
        self.norm2=norm(cout);self.film=nn.Linear(cond_dim,2*cout)
        self.conv2=nn.Conv2d(cout,cout,3,padding=1)
        self.shortcut=nn.Conv2d(cin,cout,1) if cin!=cout else nn.Identity()
        nn.init.zeros_(self.conv2.weight);nn.init.zeros_(self.conv2.bias)

    def forward(self,x,c):
        h=self.conv1(F.silu(self.norm1(x)))
        scale,shift=self.film(F.silu(c)).to(h.dtype).chunk(2,dim=1)
        h=self.norm2(h)*(1+scale[:,:,None,None])+shift[:,:,None,None]
        return self.shortcut(x)+self.conv2(F.silu(h))


class Attention(nn.Module):
    def __init__(self,channels,head_dim=64):
        super().__init__()
        if channels%head_dim:raise ValueError('Attention channels must divide head_dim')
        self.heads=channels//head_dim;self.head_dim=head_dim
        self.norm=norm(channels);self.qkv=nn.Conv2d(channels,channels*3,1)
        self.proj=nn.Conv2d(channels,channels,1)
        nn.init.zeros_(self.proj.weight);nn.init.zeros_(self.proj.bias)

    def forward(self,x):
        b,c,h,w=x.shape
        q,k,v=self.qkv(self.norm(x)).reshape(b,3,self.heads,self.head_dim,h*w).unbind(1)
        result=F.scaled_dot_product_attention(q.transpose(-1,-2),k.transpose(-1,-2),v.transpose(-1,-2))
        result=result.transpose(-1,-2).reshape(b,c,h,w)
        return x+self.proj(result)


class Stage(nn.Module):
    def __init__(self,cin,cout,attention,num_res_blocks=2):
        super().__init__()
        self.blocks=nn.ModuleList([ResBlock(cin if i==0 else cout,cout) for i in range(num_res_blocks)])
        self.attn=Attention(cout) if attention else nn.Identity()

    def forward(self,x,c):
        for block in self.blocks:x=block(x,c)
        return self.attn(x)

