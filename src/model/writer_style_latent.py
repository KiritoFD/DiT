"""Latent-only multi-scale writer style model, fresh init; no STD/script inputs."""
import torch
from torch import nn
from torch.nn import functional as F
from src.model.writer_style_pixel import Block


class WriterStyleLatent(nn.Module):
    def __init__(self,n_slots=23):
        super().__init__()
        self.stem=nn.Sequential(nn.Conv2d(4,48,3,1,1,bias=False),nn.GroupNorm(8,48),nn.SiLU())
        self.stages=nn.ModuleList([nn.Sequential(Block(48,64),Block(64,64)),
                                  nn.Sequential(Block(64,96,2),Block(96,96)),
                                  nn.Sequential(Block(96,160,2),Block(160,160))])
        self.project=nn.Sequential(nn.LayerNorm(640),nn.Linear(640,256),nn.SiLU(),nn.Linear(256,256))
        self.head=nn.Linear(256,n_slots,bias=False)
    def forward(self,z):
        if z.shape[1:]!=(4,32,32):raise ValueError('Expected canonical z0 [B,4,32,32]')
        x=self.stem(z);pooled=[]
        for stage in self.stages:
            x=stage(x);pooled.extend([x.float().mean((2,3)),x.float().var((2,3),unbiased=False).add(1e-6).sqrt()])
        e=F.normalize(self.project(torch.cat(pooled,1)),dim=-1)
        return {'embedding':e,'slot_logits':16*F.linear(e,F.normalize(self.head.weight,dim=-1))}


def build_writer_style_latent(n_slots=23,architecture='d1'):
    if architecture=='d1':return WriterStyleLatent(n_slots)
    if architecture=='geometry':
        from src.model.writer_style_geometry import WriterStyleGeometry
        return WriterStyleGeometry(n_slots)
    raise ValueError(f'Unknown latent style architecture: {architecture}')


def augment_latent(z,return_transform=False):
    b=len(z);dev=z.device
    angle=(torch.rand(b,device=dev)*2-1)*.05236;scale=.95+.1*torch.rand(b,device=dev)
    matrix=torch.zeros(b,2,3,device=dev);matrix[:,0,0]=angle.cos()*scale;matrix[:,0,1]=-angle.sin()*scale
    matrix[:,1,0]=angle.sin()*scale;matrix[:,1,1]=angle.cos()*scale;matrix[:,:,2]=(torch.rand(b,2,device=dev)*2-1)*.04
    grid=F.affine_grid(matrix,z.shape,align_corners=False)
    transformed=F.grid_sample(z,grid,align_corners=False,padding_mode='border')
    output=transformed*(.95+.1*torch.rand(b,1,1,1,device=dev))+.01*torch.randn_like(z)
    return (output,grid,scale) if return_transform else output


def warp_geometry_sdf(unclipped_sdf,grid,scale):
    # affine_grid maps output->input; geometric distances divide by its scale.
    warped=F.grid_sample(unclipped_sdf.float(),grid,align_corners=False,padding_mode='border')
    return (warped/scale[:,None,None,None]).clamp(-1,1)
