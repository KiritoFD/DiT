"""分 t 区间算 loss —— 判断高噪声区模型是否没学好。

如果高 t 的 loss 也低 -> 是我的 DDIM 采样有 bug
如果高 t 的 loss 高   -> 模型确实没学好
"""
import os
import sys

import torch

sys.path.insert(0, "/root/Workspace/xy/DiT/ref/moyi")
sys.path.insert(0, "/root/Workspace/xy/DiT/ref/moyi/moyun")
os.chdir("/root/Workspace/xy/DiT")

from dataset_moyun import MultiLabelNestedDataset  # noqa: E402
from moyun_2 import DiT_models  # noqa: E402
from utils.REPA_diffusion import create_diffusion  # noqa: E402

ck = torch.load("assets/results/moyun_repro/moyun_0025000.pt",
                map_location="cpu", weights_only=False)
m = DiT_models["moyun-12channel-B"](input_size=32, num_classes=9100,
                                    learn_sigma=True, if_rope=False)
m.load_state_dict(ck["ema"], strict=True)
m = m.eval()
diff = create_diffusion(timestep_respacing="", learn_sigma=True, device="cpu")

ds = MultiLabelNestedDataset(
    csv_file="assets/train_50k_v2_fixed.csv",
    img_shards="data/50k/shards_img",
    edge_shards="data/50k/shards_aux_canny",
    skel_shards="data/50k/shards_std_fixed", num_classes=9100)
loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False,
                                     num_workers=0)

BINS = [(0, 200), (200, 400), (400, 600), (600, 800), (800, 1000)]
acc = {b: [] for b in BINS}
acc_eps = {b: [] for b in BINS}
acc_x0 = {b: [] for b in BINS}

print("  === 分 t 区间的 loss（每区间 200 个样本）===")
with torch.no_grad():
    for bi, batch in enumerate(loader):
        if bi >= 13:
            break
        image, edge, skel, y, stroke, *_ = batch
        x0 = torch.cat([image, edge, skel], dim=1)
        y = y.squeeze(-1)
        for lo, hi in BINS:
            t = torch.randint(lo, hi, (x0.shape[0],))
            torch.manual_seed(bi * 100 + lo)
            noise = torch.randn_like(x0)
            a = diff.alphas_cumprod[t].reshape(-1, 1, 1, 1)
            x_t = a.sqrt() * x0 + (1 - a).sqrt() * noise
            out = m(x_t, t, y, stroke)
            if isinstance(out, (tuple, list)):
                out = out[0]
            eps_p, _ = out.chunk(2, dim=1)
            # eps MSE（模型训练目标）
            acc_eps[(lo, hi)].append(
                ((eps_p - noise) ** 2).mean().item())
            # x0 重建误差
            x0p = (x_t - (1 - a).sqrt() * eps_p) / a.sqrt()
            acc_x0[(lo, hi)].append((x0p - x0).abs().mean().item())

print(f"  {'t 区间':<14}{'eps_MSE':>12}{'x0_MAE':>12}")
for b in BINS:
    e = sum(acc_eps[b]) / max(len(acc_eps[b]), 1)
    x = sum(acc_x0[b]) / max(len(acc_x0[b]), 1)
    print(f"  {str(b):<14}{e:>12.4f}{x:>12.4f}")

print(f"\n  参考: 真实 x0 的 mean|.| = "
      f"{x0.abs().mean().item():.4f}；随机噪声 mean|.| ≈ 0.80")
