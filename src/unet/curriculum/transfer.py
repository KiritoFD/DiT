"""Strict stage transfer: copy every generic tensor, allow only zero Writer extensions."""
from pathlib import Path
import hashlib
import json
import torch


def initialize(model,config):
    from ..io import checkpoint_config
    path=Path(config.initialize_from);ck=torch.load(path,map_location='cpu',weights_only=False)
    old=checkpoint_config(ck)
    if old.writer_mode!='none' or config.writer_mode!='film':
        raise ValueError('Stage transfer is generic none -> Writer-only film, not legacy shared conditioning')
    for key in ('channels','num_res_blocks','decoder_res_blocks','num_writers','num_scripts',
                'vae_path','vae_scaling_factor','vae_shift_factor','std_soft_root','std_soft_sigma_min',
                'std_soft_sigma_max','std_soft_eval_sigma','act_fn','skip_gate_scales','std_gate_scales','std_shape_aug',
                'residual_cond_gate','decoder_std_residual','decoder_std_scale','stage_specific_cond'):
        a,b=getattr(old,key),getattr(config,key)
        if isinstance(a,(list,tuple)):a=tuple(a)
        if isinstance(b,(list,tuple)):b=tuple(b)
        if a!=b:raise ValueError('Transfer contract changed: '+key)
    for key in ('font_remap_json','char_remap_json'):
        if json.loads(Path(getattr(old,key)).read_text())!=json.loads(Path(getattr(config,key)).read_text()):
            raise ValueError('Condition mapping changed: '+key)
    source={k.removeprefix('_orig_mod.'):v for k,v in ck['ema' if config.initialize_state=='ema' else 'delta'].items()}
    target=model.state_dict();missing=set(target)-set(source)
    approved={k for k in target if '.writer_film.' in k}
    if missing!=approved or set(source)-set(target):raise ValueError('Unapproved stage checkpoint keys')
    for k,v in source.items():
        if target[k].shape!=v.shape:raise ValueError('Stage tensor shape changed: '+k)
    model.load_state_dict({**target,**source},strict=True)
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                source_step=int(ck['train_steps']),state=config.initialize_state,loaded=len(source),
                added=sorted(missing),new_clock=True,optimizer_reset=True,ema_reset=True)
