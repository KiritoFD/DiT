"""std skel 与 GT skel 到底对不对得上? 直接出图看。

列: [GT书法图 | std骨架解码 | GT实例骨架解码 | probe(X)解码]
"""
import os, sys, glob
import numpy as np
import torch
from PIL import Image
from diffusers.models import AutoencoderKL

os.chdir('/root/Workspace/xy/DiT')
sys.path.insert(0, os.getcwd())
VAE = 'data/pretrained/pretrained_models/sd-vae-ft-ema'
dev = 'cuda'
vae = AutoencoderKL.from_pretrained(VAE, local_files_only=True).eval().to(dev)
sc = 0.18215


@torch.no_grad()
def dec(z):
    im = vae.decode(torch.as_tensor(z).to(dev).float() / sc).sample
    return ((im.clamp(-1, 1) + 1) / 2).mean(1).float().cpu().numpy()


def idx(d):
    o = {}
    for sp in sorted(glob.glob(os.path.join(d, 'shard_*.npz'))):
        with np.load(sp) as z:
            for j, iid in enumerate(z['img_ids']):
                o[int(iid)] = (sp, j)
    return o


def get(m, iid):
    sp, j = m[iid]
    with np.load(sp) as z:
        return z['latents'][j].astype(np.float32)


di, si, ii = idx('data/top10_style23/shards_std'), \
    idx('data/top10_style23/shards_aux_skel3'), idx('data/top10_style23/shards_img')
common = sorted(set(di) & set(si) & set(ii))
rng = np.random.RandomState(3)
pick = [common[k] for k in rng.choice(len(common), 6, replace=False)]

G = np.stack([get(di, i) for i in pick])
Y = np.stack([get(si, i) for i in pick])
X = np.stack([get(ii, i) for i in pick])
gz, yz, xz = dec(G), dec(Y), dec(X)

# 直接墨迹掩膜 IoU (不骨架化)
for k, iid in enumerate(pick):
    a, b = gz[k] < 0.5, yz[k] < 0.5
    u = (a | b).sum()
    print(f"  id={iid}  std墨量={a.mean():.4f} gt墨量={b.mean():.4f} "
          f"直接掩膜IoU={float((a & b).sum() / max(u, 1)):.4f}")

rows = []
for k in range(6):
    row = np.concatenate([xz[k], gz[k], yz[k],
                          np.abs(gz[k] - yz[k])], axis=1)
    rows.append(row)
M = np.concatenate(rows, axis=0)
M = (np.clip(M, 0, 1) * 255).astype(np.uint8)
os.makedirs('/tmp/mont', exist_ok=True)
Image.fromarray(M).save('/tmp/mont/std_vs_gt_skel.png')
print('saved /tmp/mont/std_vs_gt_skel.png', M.shape)
print('列: [GT图 | std骨架 | GT实例骨架 | |std-gt| 差异]')
