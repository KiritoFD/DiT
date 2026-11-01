"""Diagnose the latent <-> GT correspondence the pair-check is complaining about."""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from src.unet.io import checkpoint_config, training_args  # noqa: E402

K = Path('assets/abc_20261031/execution/runs/A/checkpoints/0043429.pt')
ck = torch.load(K, map_location='cpu', weights_only=False)
cfg = checkpoint_config(ck)
from src.unet.data import loader  # noqa: E402

out = Path('assets/d_20261101/runs/_probe')
out.mkdir(parents=True, exist_ok=True)
args = training_args(cfg, out)
args.ckpt_every = 1000
args.epoch_steps = 1000
ds = loader(cfg, args, None, 0)[0]

rows = {int(r['img_id']): r for r in csv.DictReader(open(cfg.data_csv, encoding='utf-8-sig'))}
print('dataset len', len(ds), 'csv rows', len(rows))

from diffusers import AutoencoderKL  # noqa: E402
vae = AutoencoderKL.from_pretrained(cfg.vae_path, local_files_only=True).cuda().eval()
sf = float(cfg.vae_scaling_factor)
sh = float(cfg.vae_shift_factor)

for i in (0, 1):
    rec = ds[i]
    iid = int(rec['img_id'])
    row = rows[iid]
    p = Path(row['image_path'])
    im = Image.open(p).convert('RGB')
    x = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 127.5 - 1
    with torch.no_grad():
        lat = vae.encode(x[None].cuda()).latent_dist.mode()
    cached = rec['latent']
    print(f'idx {i} img_id {iid} path {p} size {im.size}')
    print(f'   cached latent mean {cached.mean():+.5f} std {cached.std():.5f}')
    print(f'   encoded (raw, unscaled) mean {lat.mean():+.5f} std {lat.std():.5f}')
    print(f'   encoded scaled  mean {lat.mean()*sf:+.5f} std {lat.std()*sf:.5f}')
    print(f'   |cached - encoded*sf| = {(cached.cuda() - lat*sf).abs().mean():.5f}')
    print(f'   |cached - encoded|    = {(cached.cuda() - lat).abs().mean():.5f}')
    print(f'   row aug={row.get("aug")!r} pixel_sha={row.get("pixel_sha256","")[:16]}')
