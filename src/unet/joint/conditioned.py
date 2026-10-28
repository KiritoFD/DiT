"""Explicit-context VAE decoder conditioning, preserving the original depth."""
import torch
from torch import nn
from torch.nn import functional as F


class ConditionedDecoder(nn.Module):
    def __init__(self,decoder,writers=10,scripts=3,spatial=False):
        super().__init__();self.base=decoder;self.spatial=spatial
        self.writer=nn.Embedding(writers+1,32);self.script=nn.Embedding(scripts+1,16)
        self.affine=nn.ModuleList()
        if len(decoder.up_blocks)!=4:raise ValueError('Expected SD VAE four-stage decoder')
        for block in decoder.up_blocks[1:3]:
            c=block.resnets[-1].conv2.out_channels
            if spatial:
                layer=nn.Sequential(nn.Conv2d(49,64,1),nn.SiLU(),nn.Conv2d(64,2*c,1))
                nn.init.zeros_(layer[-1].weight);nn.init.zeros_(layer[-1].bias)
            else:
                layer=nn.Conv2d(48,2*c,1)
                nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
            self.affine.append(layer)

    def forward(self,z,w,s,g):
        h=self.base.conv_in(z);h=self.base.mid_block(h,None)
        embedding=torch.cat((self.writer(w),self.script(s)),1)[:,:,None,None]
        for i,block in enumerate(self.base.up_blocks):
            h=block(h,None)
            if i in (1,2):
                cond=embedding
                if self.spatial:
                    cond=torch.cat((embedding.expand(-1,-1,*h.shape[-2:]),
                        F.interpolate(g,size=h.shape[-2:],mode='bilinear',align_corners=False)),1)
                scale,shift=self.affine[i-1](cond).chunk(2,1)
                h=h*(1+scale)+shift
        return self.base.conv_out(self.base.conv_act(self.base.conv_norm_out(h)))
