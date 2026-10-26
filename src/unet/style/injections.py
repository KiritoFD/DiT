"""I0--I3 at matched ResBlock sites. Unit buffer gain is calibrated once."""
import torch
from torch import nn
from torch.nn import functional as F


class Injection(nn.Module):
    def __init__(self,kind,channels,std_channels,representation):
        super().__init__();self.kind=kind;self.representation=representation
        self.register_buffer('gain',torch.tensor(1.));self.record=False;self.last={}
        if kind==0:self.affine=nn.Linear(64,channels*2)
        elif kind==1:
            self.local=nn.Sequential(nn.Conv2d(channels+std_channels+64,64,1),nn.SiLU(),
                                     nn.Conv2d(64,channels*2,3,padding=1))
        elif kind==2:
            self.query=nn.Conv2d(channels,128,1)
            self.kv=nn.Linear(64,256);self.output=nn.Conv2d(128,channels,1)
        elif kind==3:
            self.down=nn.Conv2d(channels,128,1,bias=False)
            self.basis=nn.Conv2d(128,128,3,padding=1,groups=8,bias=False)
            self.output=nn.Conv2d(128,channels,1,bias=False)
            self.mix=nn.Conv2d(64,8,1) if representation in (2,3) else nn.Linear(64,8)
        else:raise ValueError('Unknown injection')
        tail=self.affine if kind==0 else self.local[-1] if kind==1 else self.output
        nn.init.normal_(tail.weight,std=1e-3)
        if tail.bias is not None:nn.init.zeros_(tail.bias)

    def forward(self,h,state,std,scale):
        if self.kind==0:
            gamma,beta=self.affine(state.global_feature).to(h.dtype).chunk(2,1)
            delta=gamma[:,:,None,None]*h+beta[:,:,None,None]
            if self.record:self.last['parameter_spatial_variance']=0.
        elif self.kind==1:
            f=state.fields[scale].to(h.dtype)
            gamma,beta=self.local(torch.cat((h,std.to(h.dtype),f),1)).chunk(2,1)
            delta=gamma*h+beta
            if self.record:
                self.last['parameter_spatial_variance']=float((gamma.float().var((-1,-2)).mean()+beta.float().var((-1,-2)).mean())/2)
        elif self.kind==2:
            b,_,height,width=h.shape
            q=self.query(h).reshape(b,4,32,height*width).transpose(-1,-2)
            k,v=self.kv(state.tokens).reshape(b,8,2,4,32).permute(2,0,3,1,4).unbind(0)
            attended=F.scaled_dot_product_attention(q,k,v)
            delta=self.output(attended.transpose(-1,-2).reshape(b,128,height,width))
            if self.record:
                attn=(q.float()@k.float().transpose(-1,-2)/32**.5).softmax(-1)
                self.last['attention_entropy']=float(-(attn*attn.clamp_min(1e-9).log()).sum(-1).mean())
                self.last['attention_spatial_variance']=float(attn.var(-2).mean())
        else:
            alpha=(self.mix(state.fields[scale]) if self.representation in (2,3)
                   else self.mix(state.global_feature)[:,:,None,None]).tanh()
            latent=self.basis(self.down(h)).reshape(len(h),8,16,*h.shape[-2:])
            # Equivalent to summing eight up-projected basis outputs. Mix BEFORE
            # the 1x1 up projection to avoid an expensive [B,8,C,H,W] tensor.
            delta=self.output((latent*alpha[:,:,None]).flatten(1,2))
            if self.record:
                self.last['dynamic_alpha_rms']=float(alpha.float().square().mean().sqrt())
                self.last['parameter_spatial_variance']=float(alpha.float().var((-1,-2),unbiased=False).mean())
        delta=delta*self.gain.to(h.dtype)*state.valid[:,None,None,None].to(h.dtype)
        if self.record:
            self.last.update(base_rms=float(h.float().square().mean().sqrt()),
                delta_rms=float(delta.float().square().mean().sqrt()),
                spatial_variance=float(delta.float().var((-1,-2)).mean()),
                shared_energy=float(delta.float().mean((-1,-2)).square().mean()))
        return h+delta
