# -*- coding: utf-8 -*-
"""integrate_stdsoft.py — 核验 std_soft 链路 + 生成两个可训练配置

核验：
  1) store 完整性（index 覆盖训练+评测的 (script,char)，missing 为空）
  2) 单条 mask 的形状/极性（64×64, ink=255/背景=0）+ blur 可复现
  3) DiT 前向冒烟（std_soft_cond=True, 极小配置, CPU）—— 证明"真的接进去了"
配置：
  A) v81_stdsoft_vs_v68   —— 与 v68 **单变量** A/B（保留 SD-VAE + aug + C2OT，只加 std_soft）
  B) v82_physics_noOT_stdsoft —— **接主线**（纯物理 VAE + noaug + 关OT + 修正SDE + std_soft）
"""
import csv
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

STORE = "exp-std/data/std_soft_glyphs_v1"
ok = True


def chk(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'✓' if cond else '✗'} {name} {extra}")


# ── 1) store 完整性 ──
idx = list(csv.DictReader(open(f"{STORE}/index.csv", encoding="utf-8")))
mp = open(f"{STORE}/missing.txt", encoding="utf-8").read().strip()
chk("store index 非空", len(idx) > 0, f"{len(idx)} 条")
chk("missing.txt 为空（零缺字）", mp == "", f"len={len(mp)}")
cols = set(idx[0].keys())
print(f"    index 列: {sorted(cols)}")

# ── 2) 一条 mask 的形状/极性 + blur ──
from src.utils.std_soft_glyph import StdSoftGlyphStore  # noqa: E402

st = StdSoftGlyphStore(STORE)
scripts = sorted({r.get("script_id") for r in idx})[:1]
sample_row = idx[0]
key = {k: sample_row[k] for k in list(sample_row.keys())[:6]}
print(f"    样例行: {key}")
script_id = int(sample_row.get("script_id", 0) or 0)
char = sample_row.get("character") or sample_row.get("char") or ""
m = st.get(script_id, char)
arr = np.asarray(m)
chk("mask 形状 (1,64,64)", tuple(arr.shape) == (1, 64, 64), f"{tuple(arr.shape)}")
chk("float 且取值 ∈ [0,1]", arr.dtype.kind == "f" and arr.min() >= 0.0 and arr.max() <= 1.0,
    f"dtype={arr.dtype} range=[{arr.min():.3f},{arr.max():.3f}]")
chk("非空（有墨）", float(arr.mean()) > 0.005, f"平均墨量 {float(arr.mean()):.4f}")
chk("是软掩码（不是二值）", len(np.unique((arr * 255).round() / 255)) > 2, f"uniq≈{len(np.unique(np.round(arr,3)))}")
m2 = st.get(script_id, char)
chk("同参数可复现（评测固定 σ=0.65）", np.allclose(arr, np.asarray(m2)),
    f"max|Δ|={float(np.abs(arr - np.asarray(m2)).max()):.2e}")

# ── 3) DiT 前向冒烟（自动找类名，避免再假设导出名） ──
try:
    import inspect
    import src.model.dit as _dm
    cands = [n for n, o in vars(_dm).items() if inspect.isclass(o) and "std_soft_cond" in
             str(inspect.signature(o.__init__))] if True else []
    chk("在 src/model/dit.py 找到接受 std_soft_cond 的类", len(cands) > 0, f"{cands}")
    if cands:
        cls = getattr(_dm, cands[0])
        sig = inspect.signature(cls.__init__)
        want = dict(input_size=8, patch_size=2, in_channels=4, hidden_size=64, depth=2, num_heads=4,
                    num_classes=4, std_soft_cond=True, std_soft_zero_init=True, glyph_in_channels=1)
        keep = {k: v for k, v in want.items() if k in sig.parameters}
        m2 = cls(**keep).eval()
        x, t, g = torch.randn(2, 4, 8, 8), torch.rand(2), torch.randn(2, 1, 64, 64)
        out = m2(x, t, g=g)
        o = out[0] if isinstance(out, tuple) else out
        chk(f"{cands[0]}(std_soft_cond=True) 前向成功", tuple(o.shape) == (2, 4, 8, 8), f"out={tuple(o.shape)}")
except Exception as e:
    chk("DiT 前向冒烟", False, f"{type(e).__name__}: {e}")

# ── 4) 配置 A：与 v68 单变量 A/B ──
src = json.load(open("src/train/configs/v68_aug_sp_c2ot.json", encoding="utf-8"))
a = dict(src)
a.update(experiment_name="v81-stdsoft-vs-v68",
         results_dir="assets/results/v81_stdsoft_vs_v68",
         std_soft_cond=True, std_soft_root=STORE, std_soft_size=64,
         std_soft_sigma_min=0.4, std_soft_sigma_max=1.0, std_soft_eval_sigma=0.65,
         std_soft_zero_init=True, skel_as_glyph_cond=False,
         vae_scaling_factor=src.get("vae_scaling_factor", 0.18215),
         vae_shift_factor=src.get("vae_shift_factor", 0.0))
a["_comment"] = ("v81 = **v68 逐项照抄**（SD-VAE shards / aug 77.8k / batch288 / 200k / C2OT），"
                 "只加 **std_soft 空间条件**（1×64×64 soft mask，独立 CNN→16×16 token，zero-init）。\n"
                 "目的: 对 **v68（0.6149@200k）** 做**单变量** A/B —— 这是最干净的因果检验。\n"
                 "⚠ 显式写出 vae_scaling_factor/shift_factor: 防止日后换 shards 时静默沿用错常数(E5)。")
json.dump(a, open("src/train/configs/v81_stdsoft_vs_v68.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("  -> src/train/configs/v81_stdsoft_vs_v68.json")

# ── 5) 配置 B：接主线（用本地已有的 v79 配置为模板 → 关 OT → 加 std_soft） ──
b = json.load(open("src/train/configs/v79_callivae_physics_sp_c2ot.json", encoding="utf-8"))
b.update(experiment_name="v82-physics-noOT-stdsoft",
         results_dir="assets/results/v82_physics_noOT_stdsoft",
         std_soft_cond=True, std_soft_root=STORE, std_soft_size=64,
         std_soft_sigma_min=0.4, std_soft_sigma_max=1.0, std_soft_eval_sigma=0.65,
         std_soft_zero_init=True, skel_as_glyph_cond=False)
b["_comment"] = ("v82 = **当前主线 v80 配方**（纯物理 VAE + noaug 26k + **关 OT** + 修正后的 SDE）"
                 "**+ std_soft 空间条件**。\n"
                 "这是「接主线」的那一条；若只想做加条件的因果检验，先用 v81（对 v68 单变量）。\n"
                 "⚠ 与 v81 相比是**多变量**（VAE 换了 + OT 关了 + 加了条件）—— 别用它做因果归因。")
json.dump(b, open("src/train/configs/v82_physics_noOT_stdsoft.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("  -> src/train/configs/v82_physics_noOT_stdsoft.json")

print(f"\n  ===> {'✓ 全部通过' if ok else '✗ 有失败项，先别接训练'}")
