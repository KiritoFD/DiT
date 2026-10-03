"""决定性测试：模型能否从加噪的 x_t 一步重建 x0？

- 重建好  -> 模型学会了，是我的 DDIM 采样有 bug
- 重建差  -> 模型确实还没学好
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
sd = ck["ema"]
m = DiT_models["moyun-12channel-B"](input_size=32, num_classes=9100,
                                    learn_sigma=True, if_rope=False)
m.load_state_dict(sd, strict=True)
m = m.eval()

diff = create_diffusion(timestep_respacing="", learn_sigma=True, device="cpu")

ds = MultiLabelNestedDataset(
    csv_file="assets/train_50k_v2_fixed.csv",
    img_shards="data/50k/shards_img",
    edge_shards="data/50k/shards_aux_canny",
    skel_shards="data/50k/shards_std_fixed", num_classes=9100)
image, edge, skel, y, stroke, *_ = ds[0]
x0 = torch.cat([image, edge, skel], dim=0)[None]     # (1,12,32,32)
y = y.squeeze(-1).unsqueeze(0)                        # (1,3)
stroke = stroke.unsqueeze(0)

print(f"  x0 真实统计: mean={x0.mean():.4f} std={x0.std():.4f}")
print(f"\n  === 单步重建测试（用真实 x0 加噪 -> 模型预测 x0）===")
for t_val in (900, 700, 500, 300, 100, 50, 10):
    t = torch.tensor([t_val])
    torch.manual_seed(0)
    eps_true = torch.randn_like(x0)
    a = diff.alphas_cumprod[t_val]
    x_t = a.sqrt() * x0 + (1 - a).sqrt() * eps_true
    with torch.no_grad():
        out = m(x_t, t, y, stroke)
        if isinstance(out, (tuple, list)):
            out = out[0]
        eps_pred, _ = out.chunk(2, dim=1)
        x0_pred = ((x_t - (1 - a).sqrt() * eps_pred) / a.sqrt())
    # 只用 image 通道比（前 4）
    d_all = (x0_pred - x0).abs().mean().item()
    d_img = (x0_pred[:, :4] - x0[:, :4]).abs().mean().item()
    # eps 预测误差（前 4 通道）
    d_eps = (eps_pred[:, :4] - eps_true[:, :4]).abs().mean().item()
    print(f"    t={t_val:>4}  x0_pred.std={x0_pred.std():.4f}  "
          f"|x0_pred-x0|={d_all:.4f} (img={d_img:.4f})  "
          f"|eps_pred-eps|={d_eps:.4f}")

print(f"\n  参考: 真实 x0 的 std={x0.std():.4f}, "
      f"随机噪声 |diff| 约 {torch.randn_like(x0).abs().mean():.4f}")
