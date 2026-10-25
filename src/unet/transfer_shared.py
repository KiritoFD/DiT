"""Shared-route stage transfer: every tensor copied, new optimizer and step clock.

Used for writer-conditioned pretraining -> Top10 post-training where both stages
keep writer_mode='shared' with an identical architecture. The writer table may be
frozen in the target stage via the external-table contract; the transfer itself
only copies tensors and never resets rows silently.
"""
import hashlib
import json
from pathlib import Path
import torch


def initialize(model, config):
    from .io import checkpoint_config
    path = Path(config.initialize_from)
    ck = torch.load(path, map_location='cpu', weights_only=False)
    old = checkpoint_config(ck)
    for key in ('architecture', 'channels', 'num_res_blocks', 'decoder_res_blocks',
                'num_writers', 'num_scripts', 'writer_mode', 'writer_film_rank',
                'decouple_writer_condition', 'stage_specific_cond', 'stage_cond_mode',
                'residual_cond_gate', 'act_fn', 'brush_dim', 'reference_dim',
                'vae_path', 'vae_scaling_factor', 'vae_shift_factor', 'std_soft_root',
                'std_shape_aug', 'std_soft_sigma_min', 'std_soft_sigma_max'):
        a, b = getattr(old, key), getattr(config, key)
        if isinstance(a, (list, tuple)): a = tuple(a)
        if isinstance(b, (list, tuple)): b = tuple(b)
        if a != b: raise ValueError('Shared stage transfer changed ' + key)
    for key in ('font_remap_json', 'char_remap_json'):
        if json.loads(Path(getattr(old, key)).read_text(encoding='utf-8')) != \
           json.loads(Path(getattr(config, key)).read_text(encoding='utf-8')):
            raise ValueError('Shared stage mapping changed ' + key)
    source = {k.removeprefix('_orig_mod.'): v for k, v in
              ck['ema' if config.initialize_state == 'ema' else 'delta'].items()}
    model.load_state_dict(source, strict=True)
    if not all(torch.equal(v, source[k].to(v)) for k, v in model.state_dict().items()):
        raise ValueError('Shared stage weights were not copied exactly')
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                source_step=int(ck['train_steps']), state=config.initialize_state,
                loaded=len(source), added=[], new_clock=True, optimizer_reset=True,
                ema_reset=True, all_tensors_equal=True)
