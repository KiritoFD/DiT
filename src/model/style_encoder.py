"""Latent + neutral STD encoder. No writer label is an input to the embedding.

Contract: SD-EMA posterior.mode()*0.18215, z[B,4,32,32]; STD ink=1,
background=0, g[B,1,64,64]; explicit script IDs; normalized 256D output.
The CNN topology matches the existing critic, but training is independent.
"""
import torch
from torch import nn
from torch.nn import functional as F


class StyleEncoder(nn.Module):
    def __init__(self, n_slots, n_scripts, dim=256):
        super().__init__()
        self.z_stem = nn.Sequential(
            nn.Conv2d(4, 32, 3, 2, 1), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.GroupNorm(8, 64), nn.GELU())
        self.g_stem = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), nn.GroupNorm(4, 16), nn.GELU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.GroupNorm(4, 32), nn.GELU(),
            nn.Conv2d(32, 32, 3, 2, 1), nn.GroupNorm(4, 32), nn.GELU())
        self.trunk = nn.Sequential(
            nn.Conv2d(96, 192, 3, 1, 1), nn.GroupNorm(16, 192), nn.GELU(),
            nn.Conv2d(192, 192, 3, 2, 1), nn.GroupNorm(16, 192), nn.GELU(),
            nn.Conv2d(192, 192, 3, 1, 1), nn.GroupNorm(16, 192), nn.GELU(),
            nn.AdaptiveAvgPool2d(1))
        self.script_emb = nn.Embedding(n_scripts, 32)
        self.fuse = nn.Sequential(nn.Linear(224, 384), nn.GELU(),
                                  nn.Linear(384, dim), nn.GELU())
        self.slot_head = nn.Linear(dim, n_slots)

    def forward(self, z, g, script):
        # White=1 conversion reproduces the established critic's STD convention.
        h = self.trunk(torch.cat([self.z_stem(z), self.g_stem(1-g)], 1)).flatten(1)
        h = self.fuse(torch.cat([h, self.script_emb(script)], -1))
        return {'embedding': F.normalize(h.float(), dim=-1), 'slot_logits': self.slot_head(h)}


class LocalStatsStyleEncoder(nn.Module):
    """V2: local (11-latent-pixel receptive field) stroke statistics, not layout.

    Script labels never enter the feature. Signed embeddings and a cosine
    classifier force CE and SupCon to optimize the same normalized geometry.
    STD determines a broad pooling region; it is not concatenated into a
    content-identifying global vector. Blank STD falls back to uniform pooling.
    """
    def __init__(self, n_slots, n_scripts, dim=256):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(4,48,3,padding=1),nn.GroupNorm(8,48),nn.SiLU(),
            nn.Conv2d(48,48,3,padding=1,groups=48),nn.GroupNorm(8,48),nn.SiLU(),
            nn.Conv2d(48,96,3,stride=2,padding=1),nn.GroupNorm(8,96),nn.SiLU(),
            nn.Conv2d(96,96,3,padding=1,groups=96),nn.GroupNorm(8,96),nn.SiLU(),
            nn.Conv2d(96,128,1),nn.GroupNorm(8,128),nn.SiLU())
        self.head = nn.Sequential(nn.LayerNorm(256),nn.Linear(256,256),nn.SiLU(),
                                  nn.Dropout(.2),nn.Linear(256,dim))
        self.class_weight = nn.Parameter(torch.empty(n_slots,dim))
        nn.init.normal_(self.class_weight,std=.02)

    def forward(self,z,g,script):
        h=self.local(z)
        mask=F.adaptive_avg_pool2d(F.max_pool2d(g.float(),9,stride=1,padding=4),h.shape[-2:])
        mask=torch.where(mask.sum((2,3),keepdim=True)>1e-6,mask,torch.ones_like(mask))
        weight=mask/mask.sum((2,3),keepdim=True).clamp_min(1e-6)
        mean=(h.float()*weight).sum((2,3))
        var=((h.float()-mean[:,:,None,None]).square()*weight).sum((2,3))
        embedding=F.normalize(self.head(torch.cat([mean,(var+1e-5).sqrt()],1)).float(),dim=-1)
        logits=16*F.linear(embedding,F.normalize(self.class_weight.float(),dim=-1))
        return {'embedding':embedding,'slot_logits':logits}


def build_style_encoder(n_slots,n_scripts,arch='legacy'):
    if arch=='legacy':
        return StyleEncoder(n_slots,n_scripts)
    if arch=='local_stats':
        return LocalStatsStyleEncoder(n_slots,n_scripts)
    raise ValueError(f'Unknown style encoder architecture: {arch}')


def augment_style_inputs(z,g):
    """Mild joint affine, latent noise/gain, STD dropout. Training only.

    Both spatial inputs use the SAME affine transform. No image or cache is
    overwritten. These are encoder regularizers, not changes to DiT STD input.
    """
    n=z.shape[0]
    angle=(torch.rand(n,device=z.device)-.5)*.10
    scale=1+(torch.rand(n,device=z.device)-.5)*.08
    theta=torch.zeros(n,2,3,device=z.device)
    theta[:,0,0]=angle.cos()*scale;theta[:,0,1]=-angle.sin()*scale
    theta[:,1,0]=angle.sin()*scale;theta[:,1,1]=angle.cos()*scale
    theta[:,:,2]=(torch.rand(n,2,device=z.device)-.5)*.08
    def warp(x):
        return F.grid_sample(x,F.affine_grid(theta,x.shape,align_corners=False),
                             padding_mode='border',align_corners=False)
    z=warp(z)*(1+(torch.rand(n,1,1,1,device=z.device)-.5)*.10)+.02*torch.randn_like(z)
    g=warp(g)
    g=g*(torch.rand(n,1,1,1,device=z.device)>.15)
    return z,g

