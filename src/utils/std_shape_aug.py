"""Controlled STD-only geometry; true high-resolution raster, never GT targets.

Pixel magnitudes are defined on the64x64 condition grid. Geometry guards reject
extreme changes; they cannot certify character identity without a semantic judge.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Severity:
    rotation: float
    scale: float
    shear: float
    translation: float
    elastic: float
    width: float
    sigma: tuple


LEVELS = (Severity(0,0,0,0,0,0,(.65,.65)),
          Severity(4,.06,.04,1.,1.,.5,(.2,.8)),
          Severity(8,.12,.08,2.,2.,1.,(.4,1.2)),
          Severity(10,.16,.10,2.5,2.5,1.5,(.6,1.4)))


def mixture(step):
    if step < 10000: return (.30,.35,.30,.05)
    if step < 150000: return (.20,.25,.40,.15)
    return (.25,.30,.35,.10)


def low_frequency_field(n, h, w, amplitude, device, generator=None):
    coarse=torch.randn(n,2,4,4,device=device,generator=generator)
    field=F.interpolate(coarse,size=(h,w),mode='bicubic',align_corners=False)
    field=field-field.mean((2,3),keepdim=True)
    magnitude=field.square().sum(1,keepdim=True).sqrt().amax((2,3),keepdim=True)
    return field/magnitude.clamp_min(1e-8)*amplitude[:,None,None,None]


def variable_blur(x,sigma):
    radius=math.ceil(float(1.4*x.shape[-1]/64)*3)
    axis=torch.arange(-radius,radius+1,device=x.device,dtype=x.dtype)
    kernel=torch.exp(-axis[None].square()/(2*sigma[:,None].clamp_min(.01).square()))
    kernel=kernel/kernel.sum(1,keepdim=True)
    # One channel per sample: exact per-example Gaussian, no cross-image mixing.
    n=x.shape[0];value=x.transpose(0,1)
    value=F.conv2d(value,kernel[:,None,None,:],padding=(0,radius),groups=n)
    value=F.conv2d(value,kernel[:,None,:,None],padding=(radius,0),groups=n)
    return value.transpose(0,1)


def width_jitter(x,width):
    """Width is total stroke-width change, not radius; soften fractional HR pixels."""
    radius=width.abs()*x.shape[-1]/64/2
    out=x*(1-radius.clamp(0,1))[:,None,None,None]
    max_radius=math.ceil(1.5*x.shape[-1]/64/2)
    for r in range(1,max_radius+1):
        k=2*r+1
        dilated=F.max_pool2d(x,k,1,r)
        eroded=-F.max_pool2d(-F.pad(x,(r,r,r,r),value=0),k,1)
        value=torch.where((width>=0)[:,None,None,None],dilated,eroded)
        weight=(1-(radius-r).abs()).clamp_min(0)
        out=out+weight[:,None,None,None]*value
    return out


def guards(candidate,clean,clipped_fraction):
    total=candidate.sum((1,2,3));base=clean.sum((1,2,3)).clamp_min(1e-6)
    ratio=total/base
    similarity=2*torch.minimum(candidate,clean).sum((1,2,3))/(total+base).clamp_min(1e-6)
    edge=candidate.clone();edge[:,:,2:-2,2:-2]=0
    edge_fraction=edge.sum((1,2,3))/total.clamp_min(1e-6)
    good=torch.isfinite(candidate).flatten(1).all(1)&(ratio>=.45)&(ratio<=1.8)&(similarity>=.3)
    good=good&(clipped_fraction<=.03)&(edge_fraction<=.08)
    return good,dict(mass_ratio=ratio,soft_overlap=similarity,edge_fraction=edge_fraction,
                     clipped_fraction=clipped_fraction)


class StdShapeAugment(nn.Module):
    def __init__(self,mode='strong',attempts=2):
        super().__init__()
        if mode not in ('strong','mild'):raise ValueError(mode)
        if attempts<1:raise ValueError('At least one guarded proposal is required')
        self.mode,self.attempts=mode,attempts

    @torch.no_grad()
    def forward(self,raw,clean,step=0,*,generator=None,force_tier=None):
        if raw.ndim!=4 or raw.shape[1]!=1 or raw.shape[-2]!=raw.shape[-1] or raw.shape[-1]<256:
            raise ValueError('STD shape augmentation requires a genuine [N,1,H,H] raster, H>=256')
        if clean.shape!=(len(raw),1,64,64):raise ValueError('Clean STD must be [N,1,64,64]')
        x=raw.float()/255 if raw.dtype==torch.uint8 else raw.float()
        if raw.dtype!=torch.uint8 and (not torch.isfinite(x).all() or x.min()<0 or x.max()>1):
            raise ValueError('Invalid high-resolution ink=1 raster')
        n,_,h,w=x.shape;device=x.device
        def rand(*shape):return torch.rand(*shape,device=device,generator=generator)
        probabilities=torch.tensor(mixture(step),device=device)
        if force_tier is not None and int(force_tier) not in range(4):raise ValueError('Invalid severity tier')
        requested=torch.multinomial(probabilities,n,replacement=True,generator=generator)
        if force_tier is not None:requested=torch.full_like(requested,int(force_tier))
        tiers=torch.where(requested>0,torch.ones_like(requested),requested) if self.mode=='mild' else requested
        table=torch.tensor([[v.rotation,v.scale,v.shear,v.translation,v.elastic,v.width,*v.sigma] for v in LEVELS],device=device)
        spec=table[tiers];factor=h/64
        gy,gx=torch.meshgrid((torch.arange(h,device=device)+.5)*2/h-1,
                             (torch.arange(w,device=device)+.5)*2/w-1,indexing='ij')
        homogeneous=torch.stack((gx,gy,torch.ones_like(gx)),-1)
        result=clean.clone();accepted=(tiers==0);accepted_clip=torch.zeros(n,device=device)
        for _ in range(self.attempts):
            angle=(rand(n)*2-1)*spec[:,0]*math.pi/180
            sx=1+(rand(n)*2-1)*spec[:,1];sy=1+(rand(n)*2-1)*spec[:,1]
            shear=(rand(n)*2-1)*spec[:,2]
            tx=(rand(n)*2-1)*spec[:,3]*2/64;ty=(rand(n)*2-1)*spec[:,3]*2/64
            matrix=torch.eye(3,device=device).repeat(n,1,1)
            matrix[:,0,0]=angle.cos()*sx;matrix[:,1,0]=angle.sin()*sx
            matrix[:,0,1]=angle.cos()*shear-angle.sin()*sy
            matrix[:,1,1]=angle.sin()*shear+angle.cos()*sy
            matrix[:,0,2]=tx;matrix[:,1,2]=ty
            grid=F.affine_grid(torch.linalg.inv(matrix)[:,:2],x.shape,align_corners=False)
            field=low_frequency_field(n,h,w,spec[:,4]*factor,device,generator)
            grid=grid+field.permute(0,2,3,1)*(2/h)
            width=(rand(n)*2-1)*spec[:,5]
            proposal=F.grid_sample(width_jitter(x,width),grid,mode='bilinear',padding_mode='zeros',align_corners=False)
            sigma=(spec[:,6]+rand(n)*(spec[:,7]-spec[:,6]))*factor
            blurred=variable_blur(proposal,sigma)
            # Integer area reduction avoids the deployed CUDA adaptive-pool failure.
            proposal=(F.avg_pool2d(blurred,h//64) if h%64==0 else
                      F.interpolate(blurred,size=(64,64),mode='area'))
            proposal=(proposal*(.85+.15*rand(n))[:,None,None,None]).clamp(0,1)
            projected=torch.einsum('nij,hwj->nhwi',matrix[:,:2],homogeneous)
            outside=(projected.abs()>1).any(-1)
            clipped=(outside[:,None]*x).sum((1,2,3))/x.sum((1,2,3)).clamp_min(1e-6)
            good,accepted_stats=guards(proposal,clean,clipped)
            take=good&~accepted
            result=torch.where(take[:,None,None,None],proposal,result)
            accepted_clip=torch.where(take,clipped,accepted_clip)
            accepted=accepted|good
        fallback=~accepted
        # Clean rows and rejected candidates remain BIT-EXACT canonical evaluation STD.
        _,final_stats=guards(result,clean,accepted_clip)
        final_stats.update(requested_tier=requested,effective_tier=tiers,
            fallback=fallback,accepted=(~fallback),requested_clipped_fraction=accepted_stats['clipped_fraction'])
        return result,final_stats
