# -*- coding: utf-8 -*-
"""
_probe_flow_internals.py — 验证 flow 模式下若干训练机制的实际可用性。

要查的三件事
------------
1. FlowMatching.training_losses 是否返回 pred_xstart？
   （train.py 里 w_std_mid / w_latent_skel / w_latent_canny / latent_struct
     都依赖 loss_dict["pred_xstart"]，若 flow 下没有则全部静默失效）

2. OT-CFM（use_ot）能否正常工作、batch=192 时匈牙利算法的耗时？

3. timestep shift 的 schedule 映射方向是否符合注释描述？

不涉及训练，纯前向/数值验证，CPU 可跑。
"""
import os, sys, time, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import torch.nn.functional as F

print("=" * 74)
print("1) FlowMatching.training_losses 返回了哪些 key？")
print("=" * 74)

from src.loss.flow_matching import FlowMatching

fm = FlowMatching(num_steps=25, t_sampler="logit_normal",
                  t_mean=0.0, t_std=1.0, shift=1.0,
                  sampler="heun", heun_batch=True)


class TinyModel(torch.nn.Module):
    """最小可微替身：输入 (N,C,H,W) + t，输出同形。"""
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.w = torch.nn.Linear(10, 1)

    def forward(self, x, t, **kw):
        # 让输出依赖参数，保证有梯度
        h = x.mean().expand(1, 10) * 0.01
        return x + self.w(h.to(torch.float32)).sum() * 0.0 + x * 0.0 + torch.zeros_like(x)


torch.manual_seed(0)
B, C, H, W = 4, 4, 32, 32
m = TinyModel(C)
x0 = torch.randn(B, C, H, W)
t = fm.sample_t(B, torch.device("cpu"))

terms = fm.training_losses(m, x0, t)
print(f"  flow 返回的 keys : {sorted(terms.keys())}")
print(f"  有 pred_xstart?  : {'pred_xstart' in terms}")
if "pred_xstart" in terms:
    print(f"    shape={tuple(terms['pred_xstart'].shape)}")
else:
    print("    => train.py:1005 的 loss_dict.get('pred_xstart', None) 恒为 None")
    print("    => 依赖它的机制全部静默失效：")
    print("       w_std_mid        (去噪中段锚定标准字形 latent g)")
    print("       w_latent_skel    (latent 骨架损失)")
    print("       w_latent_canny   (latent canny 损失)")
    print("       latent_struct    (LatentStructureLoss)")

# 对照：gaussian_diffusion 是否返回 pred_xstart
try:
    from src.loss import create_diffusion_or_flow
    gd = create_diffusion_or_flow("ddpm", steps=1000)
    t_i = torch.randint(0, 1000, (B,))
    terms_d = gd.training_losses(m, x0, t_i)
    print(f"\n  对照 ddpm 返回 keys : {sorted(terms_d.keys())}")
    print(f"  ddpm 有 pred_xstart? : {'pred_xstart' in terms_d}")
except Exception as e:
    print(f"\n  [skip] ddpm 对照失败: {type(e).__name__}: {e}")


print("\n" + "=" * 74)
print("2) OT-CFM（use_ot）可用性与耗时")
print("=" * 74)
try:
    from scipy.optimize import linear_sum_assignment
    have_scipy = True
except ImportError:
    have_scipy = False
    print("  scipy 不可用 -> use_ot 无法启用")
print(f"  scipy.optimize.linear_sum_assignment 可用: {have_scipy}")

if have_scipy:
    import numpy as np
    fm_ot = FlowMatching(num_steps=25, use_ot=True, t_sampler="logit_normal")
    for bs in (32, 96, 192):
        torch.manual_seed(0)
        xb = torch.randn(bs, C, H, W)
        nb = torch.randn(bs, C, H, W)
        t0 = time.time()
        with torch.no_grad():
            xf = xb.reshape(bs, -1).float()
            nf = nb.reshape(bs, -1).float()
            cost = torch.cdist(xf, nf, p=2).pow(2)
            r, c = linear_sum_assignment(cost.numpy())
        dt = time.time() - t0
        print(f"  batch={bs:<4} 匈牙利耗时 {dt*1000:>7.1f} ms  "
              f"(cost {bs}x{bs})")
    print("  注：这是纯 CPU 耗时，每 step 一次；训练 step 本身约 300ms，")
    print("      192 批量下 OT 开销需实测占比。")


print("\n" + "=" * 74)
print("3) timestep shift 的 schedule 方向")
print("=" * 74)
print("  t = shift*s / (1 + (shift-1)*s)，s 从 1 降到 0")
print("  shift<1 应把步数推向低噪声端(t->0)，适合细节主导任务\n")
for sh in (0.6, 0.8, 1.0, 1.5, 3.0):
    f = FlowMatching(num_steps=8, shift=sh)
    sch = f._schedule(8, torch.device("cpu"))
    # 统计落在 t<0.5 的步数比例
    frac_low = float((sch < 0.5).float().mean())
    print(f"  shift={sh:<4} schedule={[round(v,3) for v in sch.tolist()]}")
    print(f"            t<0.5 的步数占比 = {frac_low*100:.1f}%  "
          f"{'(细节端密集)' if frac_low > 0.5 else '(噪声端密集)' if frac_low < 0.5 else '(均匀)'}")

print("\n  当前配置 shift=1.0（均匀）。")
print("  flow_matching.py:52 注释自己写道：『笔画末端、飞白等细节在 t→0 形成』，")
print("  而 shift<1 正是『步数向低噪声端集中，适合细节/纹理主导的任务』。")
print("  => 注释的推理方向支持 shift<1，但默认值取了 1.0，值得 ablation。")
