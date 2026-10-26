"""R0--R4: ID-only runtime style, with STD-only spatial representations."""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class StyleState:
    global_feature: torch.Tensor
    tokens: torch.Tensor
    fields: tuple
    valid: torch.Tensor


def geometry(g):
    """Continuous STD geometry. End/junction channels are soft proxies, not labels."""
    g=g.float()
    k=g.new_tensor([[-1,0,1],[-2,0,2],[-1,0,1]])[None,None]/8
    dx=F.conv2d(g,k,padding=1);dy=F.conv2d(g,k.transpose(2,3),padding=1)
    xx=F.avg_pool2d(dx.square(),5,1,2);yy=F.avg_pool2d(dy.square(),5,1,2)
    xy=F.avg_pool2d(dx*dy,5,1,2);disc=((xx-yy).square()+4*xy.square()).clamp_min(0).sqrt()
    c2=(xx-yy)/(disc+1e-5);s2=2*xy/(disc+1e-5);confidence=disc/(xx+yy+1e-5)
    mag=(dx.square()+dy.square()+1e-5).sqrt()
    curv=F.conv2d(dx/mag,k,padding=1)+F.conv2d(dy/mag,k.transpose(2,3),padding=1)
    def erode(x):return -F.max_pool2d(-x,3,1,1)
    x=g;distance=g.clone();skel=torch.zeros_like(g)
    for _ in range(8):
        opened=F.max_pool2d(erode(x),3,1,1)
        delta=(x-opened).relu();skel=skel+(delta-skel*delta).relu()
        x=erode(x);distance=distance+x
    outside=1-g;od=outside.clone()
    for _ in range(8):outside=erode(outside);od=od+outside
    width=distance/9
    wx=F.conv2d(width,k,padding=1);wy=F.conv2d(width,k.transpose(2,3),padding=1)
    neighbours=F.conv2d(skel,torch.ones_like(k),padding=1)-skel
    ends=skel*torch.exp(-(neighbours-1).square()/0.72)
    joints=skel*torch.sigmoid(4*(neighbours-2.5))
    y=torch.linspace(-1,1,g.shape[-2],device=g.device)[None,None,:,None].expand_as(g)
    x=torch.linspace(-1,1,g.shape[-1],device=g.device)[None,None,None,:].expand_as(g)
    return torch.cat((g,c2,s2,confidence,width,wx,wy,curv.tanh(),ends,joints,
                      (distance-od)/9,x,y,skel,dx,dy),1)


class TokenPool(nn.Module):
    def __init__(self):
        super().__init__();self.query=nn.Parameter(torch.randn(8,64)*.02)
    def forward(self,fields):
        context=torch.cat([F.adaptive_avg_pool2d(f,8).flatten(2).transpose(1,2) for f in fields],1)
        q=self.query[None].expand(len(context),-1,-1)
        return F.scaled_dot_product_attention(q[:,None],context[:,None],context[:,None])[:,0]


class Representation(nn.Module):
    def __init__(self,kind,writers=10,vector_dim=32):
        super().__init__();self.kind=kind;self.writers=writers
        if kind==0:
            self.table=nn.Embedding(writers+1,vector_dim);self.project=nn.Linear(vector_dim,8*64)
        elif kind in (1,4):self.table=nn.Embedding(writers+1,8*64)
        elif kind==2:
            self.table=nn.Embedding(writers+1,32);self.project=nn.Linear(32,64)
            self.local=nn.ModuleList([nn.Sequential(nn.Conv2d(c,64,3,padding=1),nn.SiLU(),
                                                    nn.Conv2d(64,64,1)) for c in (64,128,192)])
            self.pool=TokenPool()
        elif kind==3:
            self.table=nn.Embedding(writers+1,16*64)
            self.router=nn.Sequential(nn.Conv2d(16,32,3,padding=1),nn.SiLU(),nn.Conv2d(32,16,1))
            self.pool=TokenPool()
        else:raise ValueError('Unknown representation')
        nn.init.normal_(self.table.weight,std=.02)

    def forward(self,writer,pyramid,g):
        valid=(writer>=0)&(writer<self.writers)
        ids=torch.where(valid,writer,self.writers)
        vector=self.table(ids)
        if self.kind in (0,1,4):
            tokens=(self.project(vector) if self.kind==0 else vector).reshape(-1,8,64)
            tokens=F.layer_norm(tokens,(64,))
            global_feature=tokens.mean(1)
            fields=tuple(global_feature[:,:,None,None].expand(-1,-1,n,n) for n in (32,16,8))
        elif self.kind==2:
            a=self.project(vector)[:,:,None,None]
            fields=tuple(F.group_norm(f(p)*a,1) for f,p in zip(self.local,pyramid))
            tokens=self.pool(fields);global_feature=tokens.mean(1)
        else:
            bases=F.layer_norm(vector.reshape(-1,16,64),(64,))
            geom=geometry(g)
            fields=tuple(torch.einsum('bkhw,bkd->bdhw',
                         self.router(F.interpolate(geom,size=(n,n),mode='bilinear',align_corners=False)).softmax(1),bases)
                         for n in (32,16,8))
            tokens=self.pool(fields);global_feature=tokens.mean(1)
        return StyleState(global_feature,tokens,fields,valid)
