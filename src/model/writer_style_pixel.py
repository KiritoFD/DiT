"""Image-only multi-scale writer-style encoder; no STD, writer or script input."""
import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self,cin,cout,stride=1):
        super().__init__()
        self.body=nn.Sequential(nn.Conv2d(cin,cout,3,stride,1,bias=False),nn.GroupNorm(8,cout),nn.SiLU(),
                                nn.Conv2d(cout,cout,3,1,1,bias=False),nn.GroupNorm(8,cout))
        self.skip=nn.Conv2d(cin,cout,1,stride,bias=False) if cin!=cout or stride!=1 else nn.Identity()
    def forward(self,x):return F.silu(self.body(x)+self.skip(x))


class WriterStylePixel(nn.Module):
    def __init__(self,n_slots=23):
        super().__init__()
        self.stem=nn.Sequential(nn.Conv2d(1,32,5,2,2,bias=False),nn.GroupNorm(8,32),nn.SiLU())
        self.stages=nn.ModuleList([nn.Sequential(Block(32,96,2),Block(96,96)),
                                  nn.Sequential(Block(96,160,2),Block(160,160)),
                                  nn.Sequential(Block(160,256,2),Block(256,256))])
        self.project=nn.Sequential(nn.LayerNorm(2*(96+160+256)),nn.Linear(1024,384),nn.SiLU(),nn.Linear(384,256))
        self.head=nn.Linear(256,n_slots,bias=False)
    def forward(self,gray):
        x=self.stem(2*gray-1);pooled=[]
        for stage in self.stages:
            x=stage(x)
            pooled.extend([x.float().mean((2,3)),x.float().var((2,3),unbiased=False).add(1e-6).sqrt()])
        e=F.normalize(self.project(torch.cat(pooled,1)),dim=-1)
        return {'embedding':e,'slot_logits':16*F.linear(e,F.normalize(self.head.weight,dim=-1))}


def augment_gray(gray):
    """Mild affine, contrast/background/noise and blur; no morphology or flips."""
    b=len(gray);dev=gray.device
    angle=(torch.rand(b,device=dev)*2-1)*.05236;scale=.95+.10*torch.rand(b,device=dev)
    matrix=torch.zeros(b,2,3,device=dev);matrix[:,0,0]=angle.cos()*scale;matrix[:,0,1]=-angle.sin()*scale
    matrix[:,1,0]=angle.sin()*scale;matrix[:,1,1]=angle.cos()*scale;matrix[:,:,2]=(torch.rand(b,2,device=dev)*2-1)*.04
    grid=F.affine_grid(matrix,gray.shape,align_corners=False)
    ink=F.grid_sample(1-gray,grid,align_corners=False,padding_mode='zeros')
    ink=ink*(.85+.3*torch.rand(b,1,1,1,device=dev))
    # Fixed sigma .65 gaussian, selected independently per sample.
    axis=torch.arange(-2,3,device=dev).float();k=torch.exp(-axis.square()/(2*.65**2));k=k/k.sum()
    blur=F.conv2d(F.pad(ink,(2,2,2,2),mode='replicate'),(k[:,None]*k[None,:])[None,None])
    ink=torch.where(torch.rand(b,1,1,1,device=dev)<.25,blur,ink)
    return (1-ink-.03*torch.rand(b,1,1,1,device=dev)+.005*torch.randn_like(gray)).clamp(0,1)
