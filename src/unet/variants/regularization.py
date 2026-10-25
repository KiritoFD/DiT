"""Small opt-in regularizers; default path preserves historical training semantics."""
import torch
from torch import nn


def schedule_scale(step,start=2000,end=5000):
    if not 0 <= start < end:raise ValueError('Invalid DropPath ramp')
    return min(1.,max(0.,(step-start)/(end-start)))


def drop_residual(x,probability,scale=None,training=True):
    if not training or probability==0:return x
    if not 0<=probability<1:raise ValueError('DropPath probability must be in [0,1)')
    keep=1-probability*(1. if scale is None else scale)
    mask=(torch.rand((x.shape[0],)+(1,)*(x.ndim-1),device=x.device)<keep).to(x.dtype)
    return x*mask/keep


def optimizer_parameters(model,config):
    if config.writer_mode=='film':
        from ..curriculum.runtime import groups
        return groups(model,config)
    cond_lr=getattr(config,'writer_cond_lr',0.)
    if cond_lr and config.writer_mode=='shared':
        cond_names=('writer.','writer_proj.','script.')
        cond=[];back=[]
        for n,p_ in model.named_parameters():
            if not p_.requires_grad:continue
            (cond if n.startswith(cond_names) else back).append(p_)
        return [dict(params=back,lr=config.lr),
                dict(params=cond,lr=cond_lr)]
    if not config.grouped_weight_decay:return model.parameters()
    embedding={id(module.weight) for module in model.modules() if isinstance(module,nn.Embedding)}
    decay=[];no_decay=[]
    for _,p in model.named_parameters():
        if not p.requires_grad:continue
        (no_decay if p.ndim<=1 or id(p) in embedding else decay).append(p)
    return [dict(params=decay,weight_decay=config.weight_decay),dict(params=no_decay,weight_decay=0.)]
