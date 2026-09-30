import numpy as np, torch, glob, os
import torch.nn.functional as F
from diffusers.models import AutoencoderKL

dev = 'cuda'
vae = AutoencoderKL.from_pretrained('data/pretrained/pretrained_models/sd-vae-ft-ema',
                                    local_files_only=True).eval().to(dev)
sc = 0.18215


def enc(x):
    with torch.no_grad():
        return vae.encode(x.to(dev).repeat(1, 3, 1, 1)).latent_dist.mean * sc


def dec(z):
    with torch.no_grad():
        im = vae.decode(z.to(dev).float() / sc).sample
    return ((im.clamp(-1, 1) + 1) / 2).mean(1).float().cpu()


zbg = enc(torch.ones(1, 1, 256, 256)).float().cpu()


def get(d, n=64):
    with np.load(sorted(glob.glob(os.path.join(d, 'shard_*.npz')))[0]) as z:
        return torch.from_numpy(z['latents'][:n].astype(np.float32))


print(f"z_bg: mean={float(zbg.mean()):+.4f} std={float(zbg.std()):.4f} "
      f"per-ch={[round(float(v), 3) for v in zbg.mean((0, 2, 3))]}")
print(f"      decode(z_bg) 灰度 min/max/mean = "
      f"{float(dec(zbg).min()):.3f}/{float(dec(zbg).max()):.3f}/{float(dec(zbg).mean()):.3f}")
for nm, d in [('aux_skel3(骨架)', 'data/top10_style23/shards_aux_skel3'),
              ('std(标准骨架)', 'data/top10_style23/shards_std'),
              ('img(GT图)   ', 'data/top10_style23/shards_img')]:
    Y = get(d)
    mY = Y.mean(0, keepdim=True).expand_as(Y)
    g = dec(Y)
    print(f"{nm} mean={float(Y.mean()):+.4f} std={float(Y.std()):.4f} "
          f"|ch_mean|={float(Y.mean((0, 2, 3)).abs().mean()):.4f} "
          f"| MSE(zbg,Y)={float(F.mse_loss(zbg.expand_as(Y), Y)):.4f} "
          f"MSE(meanY,Y)={float(F.mse_loss(mY, Y)):.4f} "
          f"| 解码墨量={float((g < 0.5).float().mean()):.4f} "
          f"灰度mean={float(g.mean()):.3f}")
