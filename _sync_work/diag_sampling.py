"""定位采样 bug：分别测 (a) 去掉 clamp (b) 从不同 t 起采样。"""
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
image, edge, skel, y, stroke, *_ = ds[0]
x0_true = torch.cat([image, edge, skel], dim=0)[None]
y = y.squeeze(-1).unsqueeze(0)
stroke = stroke.unsqueeze(0)
print(f"  x0_true: std={x0_true.std():.4f}\n")


def ddim(model, x_start, t_start, steps=50, cfg=1.0, use_clamp=True):
    """从 t_start 采样到 0。x_start 是 t_start 处的 x。

    ⚠ forward_with_cfg 内部 half = x[:len(x)//2] -> batch 必须是**真实 batch 的 2 倍**
    """
    B = x_start.shape[0]
    x = torch.cat([x_start, x_start], dim=0)          # 2B
    y2 = torch.cat([y, y], dim=0)
    s2 = torch.cat([stroke, stroke], dim=0)
    ts = torch.linspace(t_start, 0, steps, device="cpu").long()
    for i, t in enumerate(ts):
        tb = t.expand(x.shape[0])
        e = model.forward_with_cfg(x, tb, y2, s2, cfg)[: x.shape[0]]
        e, _ = e.chunk(2, dim=1)
        a = diff.alphas_cumprod[t].reshape(-1, 1, 1, 1)
        x0 = (x - (1 - a).sqrt() * e) / a.sqrt()
        if use_clamp:
            x0 = x0.clamp(-4, 4)
        if i + 1 < len(ts):
            an = diff.alphas_cumprod[ts[i + 1]].reshape(-1, 1, 1, 1)
            x = an.sqrt() * x0 + (1 - an).sqrt() * e
        else:
            x = x0
    return x


print("  === (a) 从真实 x0 加噪到各 t，再采样回 0 ===")
print(f"  {'t_start':<10}{'clamp':<8}{'out.std':>10}{'|out-x0|':>12}")
for t_start in (900, 700, 500, 300):
    a = diff.alphas_cumprod[t_start]
    torch.manual_seed(0)
    x_t = a.sqrt() * x0_true + (1 - a).sqrt() * torch.randn_like(x0_true)
    for uc in (True, False):
        out = ddim(m, x_t, t_start, steps=50, cfg=1.0, use_clamp=uc)
        d = (out - x0_true).abs().mean().item()
        print(f"  {t_start:<10}{str(uc):<8}{out.std():>10.4f}{d:>12.4f}")

print(f"\n  === (b) 纯噪声起（cfg=1.0，标准 DDIM）===")
torch.manual_seed(0)
out = ddim(m, torch.randn(1, 12, 32, 32), 999, steps=50, cfg=1.0)
print(f"    out.std={out.std():.4f}  |out-x0|={(out-x0_true).abs().mean():.4f}"
      f"  (真实 std={x0_true.std():.4f})")
