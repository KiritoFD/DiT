"""Two opt-in, zero-initialized spatial priors; no changed inference conditions."""
import torch
from torch import nn


class SpatialProbe(nn.Module):
    def __init__(self, mode, width, depth, channels):
        super().__init__()
        if mode not in ('long-skip', 'local-residual'):
            raise ValueError(mode)
        self.mode = mode
        self.source = depth//4
        self.target = 3*depth//4
        if self.source >= self.target:
            raise ValueError('Spatial skip requires at least two blocks')
        if mode == 'long-skip':
            self.norm = nn.LayerNorm(width)
            self.proj = nn.Linear(width, width)
            nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)
        else:
            self.local = nn.Sequential(nn.Conv2d(channels, 32, 3, padding=1),
                                       nn.SiLU(), nn.Conv2d(32, channels, 3, padding=1))
            nn.init.zeros_(self.local[-1].weight); nn.init.zeros_(self.local[-1].bias)

    def skip(self, x, shallow):
        return x + self.proj(self.norm(shallow))

    def refine(self, velocity):
        return velocity + self.local(velocity)


def attach_spatial_probe(model, mode):
    """Attach *after* loading a baseline for warm-start, before optimizer/EMA.

    build_dit_from_args attaches before strict loading when restoring a trained
    probe checkpoint. This function never relaxes strict state_dict checks.
    """
    if mode == 'off': return model
    if getattr(model, 'spatial_probe', None) is not None:
        raise ValueError('Spatial probe already attached')
    model.spatial_probe = SpatialProbe(mode, model.pos_embed.shape[-1],
                                       len(model.blocks), model.out_channels)
    model.spatial_probe.to(device=model.pos_embed.device, dtype=model.pos_embed.dtype)
    return model


def prepare_checkpoint(source, mode, destination):
    """Create a weights-only EMA warm start; never reuse optimizer/step state."""
    import copy
    import hashlib
    from pathlib import Path
    from src.eval.model_io import load_model_from_ckpt
    model, args = load_model_from_ckpt(source, device='cpu', use_ema=True)
    attach_spatial_probe(model, mode)
    args = copy.deepcopy(args); args.spatial_probe = mode
    digest = hashlib.sha256()
    with open(source, 'rb') as f:
        for block in iter(lambda: f.read(2**20), b''): digest.update(block)
    state = model.state_dict()
    result = {'delta':state, 'ema':state, 'args':args, 'train_steps':0,
              'warm_start_parent': {'path':str(source), 'sha256':digest.hexdigest(),
                                    'weights':'ema', 'optimizer_restored':False}}
    with Path(destination).open('xb') as f: torch.save(result, f)


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--mode',choices=('long-skip','local-residual'),required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args();prepare_checkpoint(a.checkpoint,a.mode,a.out)
