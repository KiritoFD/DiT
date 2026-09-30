"""核对: 骨架 latent 空间到底有没有「空白比 g_std 更优」的退化吸引子。

早前两个测量互相矛盾 (MSE(zbg,Y)=0.305 vs 0.512)，必须按 shard 拆开看。
"""
import numpy as np, torch, glob, os
import torch.nn.functional as F

SH = 'data/top10_style23/shards_aux_skel3'
STD = 'data/top10_style23/shards_std'
IMG = 'data/top10_style23/shards_img'


def load_all(d):
    out, ids = [], []
    for sp in sorted(glob.glob(os.path.join(d, 'shard_*.npz'))):
        with np.load(sp) as z:
            out.append(z['latents'].astype(np.float32))
            ids.append(z['img_ids'])
    return np.concatenate(out), np.concatenate(ids)


Y, yid = load_all(SH)
G, gid = load_all(STD)
X, xid = load_all(IMG)
print(f"shapes Y={Y.shape} G={G.shape} X={X.shape}")

# 背景参考 (白底) —— 用实际白图的 latent
import torch
from diffusers.models import AutoencoderKL
dev = 'cuda'
vae = AutoencoderKL.from_pretrained('data/pretrained/pretrained_models/sd-vae-ft-ema',
                                    local_files_only=True).eval().to(dev)
with torch.no_grad():
    zbg = (vae.encode(torch.ones(1, 3, 256, 256, device=dev)).latent_dist.mean * 0.18215).float().cpu()
print(f"z_bg per-ch mean = {[round(float(v), 3) for v in zbg.mean((0, 2, 3))]}")

# 对齐 g_std 到 Y 的顺序
gmap = {int(i): k for k, i in enumerate(gid)}
sel = [gmap[int(i)] for i in yid]
G = G[sel]
print(f"g_std 对齐后 {G.shape}")

Yt, Gt = torch.from_numpy(Y), torch.from_numpy(G)
zbe = zbg.expand_as(Yt)
mY = Yt.mean(0, keepdim=True).expand_as(Yt)

print("\n=== 全局 (n=%d) ===" % len(Yt))
print(f"  MSE(zbg, Y)   = {float(F.mse_loss(zbe, Yt)):.5f}")
print(f"  MSE(meanY, Y) = {float(F.mse_loss(mY, Yt)):.5f}")
print(f"  MSE(g_std, Y) = {float(F.mse_loss(Gt, Yt)):.5f}")
print(f"  L1(zbg, Y)    = {float(F.l1_loss(zbe, Yt)):.5f}")
print(f"  L1(meanY, Y)  = {float(F.l1_loss(mY, Yt)):.5f}")
print(f"  L1(g_std, Y)  = {float(F.l1_loss(Gt, Yt)):.5f}")
print(f"  ★ MSE: 空白{'更优 ✗' if float(F.mse_loss(zbe,Yt))<float(F.mse_loss(Gt,Yt)) else '更差 ✓'}"
      f"   L1: 空白{'更优 ✗' if float(F.l1_loss(zbe,Yt))<float(F.l1_loss(Gt,Yt)) else '更差 ✓'}")

print("\n=== 逐 shard ===")
print(f"  {'shard':>6}{'n':>6}{'MSE(zbg,Y)':>13}{'MSE(meanY,Y)':>14}{'MSE(g_std,Y)':>14}{'Y.mean':>9}{'Y.std':>8}")
off = 0
for si, sp in enumerate(sorted(glob.glob(os.path.join(SH, 'shard_*.npz')))):
    with np.load(sp) as z:
        n = len(z['img_ids'])
    sl = slice(off, off + n)
    ys, gs = Yt[sl], Gt[sl]
    print(f"  {si:>6}{n:>6}{float(F.mse_loss(zbe[:n], ys)):>13.5f}"
          f"{float(F.mse_loss(mY[:n], ys)):>14.5f}{float(F.mse_loss(gs, ys)):>14.5f}"
          f"{float(ys.mean()):>9.4f}{float(ys.std()):>8.4f}")
    off += n
print(f"  合计 {off}")

# 关键: 全量下的 per-sample 谁赢
d_z = (zbe - Yt).pow(2).mean((1, 2, 3))
d_g = (Gt - Yt).pow(2).mean((1, 2, 3))
print(f"\n  per-sample MSE: zbg 赢 {(d_z < d_g).float().mean() * 100:.1f}% 的样本; "
      f"g_std 赢 {(d_g < d_z).float().mean() * 100:.1f}%")
print(f"  差值 (g_std - zbg) 的均值 = {float((d_g - d_z).mean()):+.5f}")
