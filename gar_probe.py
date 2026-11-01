"""Probe the official latent+STD dataset contract for the GAR decoder run."""
import sys, torch
from src.unet.io import checkpoint_config, training_args, load
from src.unet.data import loader

CK = "assets/abc_20261031/execution/runs/A/checkpoints/0043429.pt"
ck = torch.load(CK, map_location="cpu", weights_only=False)
cfg = checkpoint_config(ck)
print("std_shape_aug=", getattr(cfg, "std_shape_aug", None))
print("std_soft_root=", getattr(cfg, "std_soft_root", None))
print("writers=", cfg.num_writers, "scripts=", cfg.num_scripts, "workers=", getattr(cfg, "workers", None))
print("writer_mode=", getattr(cfg, "writer_mode", None))
from pathlib import Path
out = Path("assets/gar_20261101"); out.mkdir(parents=True, exist_ok=True)
args = training_args(cfg, out)
args.ckpt_every = 1000; args.epoch_steps = 1000
r = loader(cfg, args, None, 0)
print("loader returned", len(r), "values")
ds = r[0]
print("dataset len=", len(ds), "type=", type(ds).__name__)
b = ds[0]
for k, v in b.items():
    print(f"  {k}: {type(v).__name__} {getattr(v,'shape',None)} {getattr(v,'dtype',None)}")
