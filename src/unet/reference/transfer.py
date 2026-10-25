"""Reference FM stage transfer: all tensors copied, new optimizer and step clock."""
import hashlib
import json
from pathlib import Path
import torch


def initialize(model,config):
    from ..io import checkpoint_config
    path=Path(config.initialize_from)
    ck=torch.load(path,map_location='cpu',weights_only=False)
    old=checkpoint_config(ck)
    for key in ('architecture','channels','num_res_blocks','decoder_res_blocks','num_writers','num_scripts',
                'writer_mode','reference_dim','reference_max_refs','brush_dim','writer_film_rank',
                'decouple_writer_condition','stage_specific_cond','act_fn','vae_path',
                'vae_scaling_factor','vae_shift_factor','std_soft_root','std_shape_aug',
                'std_soft_sigma_min','std_soft_sigma_max'):
        # std_soft_eval_sigma excluded: evaluation-only knob differing by design.
        a,b=getattr(old,key),getattr(config,key)
        if isinstance(a,(list,tuple)):a=tuple(a)
        if isinstance(b,(list,tuple)):b=tuple(b)
        if a!=b:raise ValueError('Reference stage transfer changed '+key)
    for key in ('font_remap_json','char_remap_json'):
        if json.loads(Path(getattr(old,key)).read_text())!=json.loads(Path(getattr(config,key)).read_text()):
            raise ValueError('Reference stage mapping changed '+key)
    source={k.removeprefix('_orig_mod.'):v for k,v in ck['ema' if config.initialize_state=='ema' else 'delta'].items()}
    model.load_state_dict(source,strict=True)
    if not all(torch.equal(v,source[k].to(v)) for k,v in model.state_dict().items()):
        raise ValueError('Reference stage weights were not copied exactly')
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                source_step=int(ck['train_steps']),state=config.initialize_state,loaded=len(source),
                added=[],new_clock=True,optimizer_reset=True,ema_reset=True,all_tensors_equal=True)
