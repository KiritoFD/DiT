"""Plain convolutional U-Net components; independent of src.model.DiT."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .variants.modulation import bounded_gain, zero_linear


def norm(channels):
    return nn.GroupNorm(math.gcd(channels, 32), channels, eps=1e-6)


def get_activation(act: str = 'silu'):
    if act == 'silu': return F.silu
    if act == 'gelu': return F.gelu
    raise ValueError(f'Unsupported activation: {act}')


def timestep_embedding(t, dim=128):
    freq=torch.exp(-math.log(10000)*torch.arange(dim//2,device=t.device)/ (dim//2))
    phase=t.float()[:,None]*freq[None]
    return torch.cat([phase.cos(),phase.sin()],dim=-1)


class ResBlock(nn.Module):
    """Shared AdaGN plus optional Writer-only affine and conditional residual gain.

    Historical `film`/`writer_film` names are retained as checkpoint keys.
    """
    def __init__(self, cin, cout, cond_dim=512, act_fn='silu'):
        super().__init__()
        self.act=get_activation(act_fn)
        self.norm1=norm(cin);self.conv1=nn.Conv2d(cin,cout,3,padding=1)
        self.norm2=norm(cout);self.film=nn.Linear(cond_dim,2*cout)
        self.conv2=nn.Conv2d(cout,cout,3,padding=1)
        self.shortcut=nn.Conv2d(cin,cout,1) if cin!=cout else nn.Identity()
        self.writer_film=None
        self.residual_gate=None
        nn.init.zeros_(self.conv2.weight);nn.init.zeros_(self.conv2.bias)

    def enable_writer_film(self,rank):
        if self.writer_film is not None:raise ValueError('Writer FiLM already enabled')
        self.writer_film=nn.Sequential(nn.Linear(32,rank),nn.SiLU(),
                                       zero_linear(rank,2*self.conv2.in_channels))

    def enable_residual_gate(self):
        """Conditioned channel gains; zero logits mean gain=1, never a dead gate."""
        if self.residual_gate is not None:raise ValueError('Residual gate already enabled')
        self.residual_gate=zero_linear(self.film.in_features,self.conv2.out_channels)

    def forward(self,x,c,writer=None):
        h=self.conv1(self.act(self.norm1(x)))
        condition=self.act(c)
        scale,shift=self.film(condition).to(h.dtype).chunk(2,dim=1)
        if self.writer_film is not None:
            if writer is None:raise ValueError('Writer embedding required for independent FiLM')
            writer_scale,writer_shift=self.writer_film(writer).to(h.dtype).chunk(2,dim=1)
            scale=scale+writer_scale;shift=shift+writer_shift
        h=self.norm2(h)*(1+scale[:,:,None,None])+shift[:,:,None,None]
        residual=self.conv2(self.act(h))
        if self.residual_gate is not None:
            gain=bounded_gain(self.residual_gate(condition).to(residual.dtype))
            residual=residual*gain[:,:,None,None]
        return self.shortcut(x)+residual


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
    def __init__(self,cin,cout,attention,num_res_blocks=2,act_fn='silu'):
        super().__init__()
        self.blocks=nn.ModuleList([ResBlock(cin if i==0 else cout,cout,act_fn=act_fn) for i in range(num_res_blocks)])
        self.attn=Attention(cout) if attention else nn.Identity()

    def forward(self,x,c,writer=None):
        for block in self.blocks:x=block(x,c,writer)
        return self.attn(x)

