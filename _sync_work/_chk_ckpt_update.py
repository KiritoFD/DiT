# -*- coding: utf-8 -*-
"""_chk_ckpt_update.py — 检查冒烟 ckpt: 新通路参数是否真的更新 (远程跑)。

判据:
  1) glyph_injections.*.out_proj : zero-init 初始为 0 -> 训练后应 != 0 (通路打开)
  2) style_proj / style_role    : 风格通路参数被更新
  3) 各层 out_proj 更新幅度是否健康(不爆炸/不呆滞)
"""
import glob
import numpy as np
import torch

pat = "/root/Workspace/xy/DiT/assets/results/smoke_c41x_sty32/*/checkpoints/*.pt"
files = sorted(glob.glob(pat))
if not files:
    print("no ckpt")
    raise SystemExit(0)
p = files[-1]
print(f"ckpt: {p}")
d = torch.load(p, map_location="cpu")
sd = d.get("delta", d.get("model", d.get("model_state_dict", d)))
if not isinstance(sd, dict):
    print("unexpected ckpt format:", type(d))
    raise SystemExit(0)

print("top-level keys:", list(d.keys())[:12] if isinstance(d, dict) else type(d))
_allk = list(sd.keys())
print("n_state_keys:", len(_allk))
print("sample keys:", _allk[:8])
print("inject-ish:", [k for k in _allk if "inject" in k.lower()][:6])
print("style-ish:", [k for k in _allk if "style" in k.lower()][:6])


def stat(t):
    t = t.detach().float().numpy()
    return float(np.abs(t).mean()), float(np.abs(t).max())


print("\n=== 注入层 out_proj (zero-init -> 应 != 0) ===")
for k in sorted(k for k in sd if "out_proj" in k and "injections" in k):
    m, mx = stat(sd[k])
    print(f"  {k:<58} |mean|={m:.3e}  max={mx:.3e}")

print("\n=== 风格通路 ===")
for k in sorted(k for k in sd if "style_proj" in k or "style_role" in k):
    m, mx = stat(sd[k])
    print(f"  {k:<58} |mean|={m:.3e}  max={mx:.3e}")

print("\n=== 对照: 主干 block (确认训练整体有效) ===")
ks = [k for k in sd if "blocks.0." in k and k.endswith("weight")]
for k in sorted(ks)[:3]:
    m, mx = stat(sd[k])
    print(f"  {k:<58} |mean|={m:.3e}  max={mx:.3e}")

print("\n=== glyph_embedder ===")
for k in sorted(k for k in sd if "glyph_embedder" in k and k.endswith("weight"))[:3]:
    m, mx = stat(sd[k])
    print(f"  {k:<58} |mean|={m:.3e}  max={mx:.3e}")

n_inj = len([k for k in sd if "injections" in k and "out_proj.weight" in k])
print(f"\n注入层数: {n_inj}")
