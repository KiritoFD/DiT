"""D2/D3: spatial geometry and locally pooled brush, latent input only."""
import torch
from torch import nn
from torch.nn import functional as F
from src.model.writer_style_pixel import Block


class WriterStyleGeometry(nn.Module):
    def __init__(self,n_slots=23):
        super().__init__()
        self.stem=nn.Sequential(nn.Conv2d(4,48,3,1,1,bias=False),nn.GroupNorm(8,48),nn.SiLU())
        self.geometry=nn.ModuleList([nn.Sequential(Block(48,48),Block(48,48)),
            nn.Sequential(Block(48,64,2),Block(64,64)),nn.Sequential(Block(64,80,2),Block(80,80))])
        self.spatial=nn.ModuleList([nn.Conv2d(c,16,1) for c in (48,64,80)])
        self.geometry_project=nn.Sequential(nn.LayerNorm(768),nn.Linear(768,128),nn.SiLU())
        self.brush=nn.ModuleList([nn.Sequential(Block(48,48),Block(48,48)),
            nn.Sequential(Block(48,80,2),Block(80,80)),nn.Sequential(Block(80,128,2),Block(128,128))])
        self.brush_project=nn.Sequential(nn.LayerNorm(512),nn.Linear(512,128),nn.SiLU())
        self.project=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,256),nn.SiLU(),nn.Linear(256,256))
        self.head=nn.Linear(256,n_slots,bias=False)
        # Present in BOTH arms for exact initialization/parameter parity.
        self.sdf_head=nn.Sequential(nn.Conv2d(192,32,3,1,1),nn.SiLU(),nn.Conv2d(32,1,1),nn.Tanh())

    def forward(self,z):
        if z.shape[1:]!=(4,32,32):raise ValueError('Expected canonical z0 [B,4,32,32]')
        shared=self.stem(z);x=shared;maps=[];spatial=[]
        for stage,compress in zip(self.geometry,self.spatial):
            x=stage(x);maps.append(x)
            # 4x4 bins deliberately retain coarse spatial arrangements.
            spatial.append(F.adaptive_avg_pool2d(compress(x),4).flatten(1))
        geometry=self.geometry_project(torch.cat(spatial,1))
        x=shared;pooled=[]
        for stage in self.brush:
            x=stage(x);pooled.extend((x.float().mean((2,3)),x.float().var((2,3),unbiased=False).add(1e-6).sqrt()))
        brush=self.brush_project(torch.cat(pooled,1))
        embedding=F.normalize(self.project(torch.cat((geometry,brush),1)),dim=-1)
        full=torch.cat([maps[0]]+[F.interpolate(m,size=(32,32),mode='bilinear',align_corners=False) for m in maps[1:]],1)
        return {'embedding':embedding,'slot_logits':16*F.linear(embedding,F.normalize(self.head.weight,dim=-1)),
                'geometry_sdf':self.sdf_head(full),'geometry_shapes':tuple(m.shape[-2:] for m in maps)}
