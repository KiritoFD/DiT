"""诊断 moyun 复现：为什么 loss 低但生成是噪声。"""
import os
import sys

import torch

sys.path.insert(0, "/root/Workspace/xy/DiT/ref/moyi")
sys.path.insert(0, "/root/Workspace/xy/DiT/ref/moyi/moyun")
os.chdir("/root/Workspace/xy/DiT")

from moyun_2 import DiT_models  # noqa: E402
from dataset_moyun import MultiLabelNestedDataset  # noqa: E402
from utils.REPA_diffusion import create_diffusion  # noqa: E402

CK = "assets/results/moyun_repro/moyun_0025000.pt"
ck = torch.load(CK, map_location="cpu", weights_only=False)
print("  ckpt keys:", list(ck.keys()))
sd = ck.get("ema") or ck.get("model")
print(f"  sd 键数: {len(sd)}")

m = DiT_models["moyun-12channel-B"](input_size=32, num_classes=9100,
                                    learn_sigma=True, if_rope=False)
ms, us = m.load_state_dict(sd, strict=False)
print(f"  missing={len(ms)}  unexpected={len(us)}")
if ms:
    print(f"    missing 前5: {list(ms)[:5]}")
if us:
    print(f"    unexpected 前5: {list(us)[:5]}")

# ── 关键测试：用训练的方式算一次 loss，看是否真的低 ────────────────
ds = MultiLabelNestedDataset(
    csv_file="assets/train_50k_v2_fixed.csv",
    img_shards="data/50k/shards_img",
    edge_shards="data/50k/shards_aux_canny",
    skel_shards="data/50k/shards_std_fixed", num_classes=9100)
loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False,
                                     num_workers=0)
batch = next(iter(loader))
image, edge, skel, y, stroke, *_ = batch
x = torch.cat([image, edge, skel], dim=1)
y = y.squeeze(-1)
stroke = stroke
print(f"\n  x shape: {tuple(x.shape)}  y: {tuple(y.shape)}")

m.eval()
diff = create_diffusion(timestep_respacing="", learn_sigma=True,
                        device="cpu")
with torch.no_grad():
    t = torch.randint(0, 1000, (x.shape[0],))
    out = diff.training_losses(m, x, t, None, {"y": y, "stroke": stroke})
    print(f"  训练口径 loss = {out['loss'].mean().item():.4f}")

# ── 检查 x 的统计（训练目标）──────────────────────────────────────
print(f"\n  x (12ch) 统计: mean={x.mean():.4f} std={x.std():.4f}")
for i, nm in enumerate(["img", "edge", "skel"]):
    s = x[:, i * 4:(i + 1) * 4]
    print(f"    {nm}: mean={s.mean():.4f} std={s.std():.4f}")

# ── 检查 DDIM 采样的中间状态 ─────────────────────────────────────
print("\n  DDIM 采样中间检查（t=999 起步，看 x 是否爆炸/归零）:")
shape = (2, 12, 32, 32)
yy = torch.tensor([[591, 3, 5], [591, 3, 5]])
ss = torch.tensor([0, 0])
x_t = torch.randn(shape)
with torch.no_grad():
    for k, tt in enumerate([999, 800, 600, 400, 200, 50, 0]):
        tb = torch.full((2,), tt, dtype=torch.long)
        e = m.forward_with_cfg(x_t, tb, yy, ss, 4.0)[:2]
        e, _ = e.chunk(2, dim=1)
        a = diff.alphas_cumprod[tt]
        x0 = ((x_t - (1 - a).sqrt() * e) / a.sqrt()).clamp(-4, 4)
        print(f"    t={tt:>4}  x_t.std={x_t.std():.4f}  eps.std={e.std():.4f} "
              f" x0.std={x0.std():.4f}")
        if tt > 0:
            tn = [999, 800, 600, 400, 200, 50, 0][k + 1]
            an = diff.alphas_cumprod[tn]
            x_t = an.sqrt() * x0 + (1 - an).sqrt() * e
