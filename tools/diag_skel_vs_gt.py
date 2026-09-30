#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/diag_skel_vs_gt.py — 决定性验证: SkelNet 的形变是否真的让骨架更接近真实书法骨架?
对 strict 84 条:
  S_std  = 标准字骨架 (VAE decode data/50k_v2_glyph15k/shards_std)   <- strict 应走 eval dir
  S_pred = SkelNet(S_std, e_style)  (top10 表)
  S_gt   = 真实书法骨架 (从 strict 的 image 细化得到 或 std_path)
判据: ssim(S_pred, S_gt) > ssim(S_std, S_gt) 的比例 / 平均提升
"""
import os, sys
import numpy as np
import pandas as pd
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)
torch.set_num_threads(4)

from src.model.deform_skel import DeformSkel
from src.utils.callig_script_map import load_callig_script_map

emb = torch.load("assets/callig_script_emb_top10.pt", map_location="cpu",
                 weights_only=False)["embedding"].float()
sd = torch.load("assets/deform_skel_top10_v1.pt", map_location="cpu", weights_only=False)
sd = sd.get("deform", sd)

sk = DeformSkel(cond_dim=128, ch=4, grid=32, style_ch=32, width=96,
                max_off=6.0, coarse=8, residual=0, res_cap=1.0,
                stroke_mod=1, stroke_cap=1.0, gate_radius=0.25, dt_ch=1)
sk.load_state_dict(sd, strict=False)
sk.eval()

csmap = load_callig_script_map("assets/callig_script_id_map_top10.json")

# strict 84 的骨架: 用 eval_skel_latent_shards_dir = data/50k_v2_glyph15k/shards_std
# 这些行有 old_50k_id, 即骨架分片的 idx
df = pd.read_csv("assets/eval_top10_strict_subset84.csv")
print("strict 84 列:", list(df.columns))
print(df.head(3).to_string())

# GT 骨架直接用 csv 的 std_path (即真实书法里提取的 50k std)
def load_gray(p, size=128):
    from PIL import Image
    return np.asarray(Image.open(p).convert("L").resize((size, size),
                      Image.Resampling.LANCZOS)).astype(np.float32) / 255.0

from skimage.metrics import structural_similarity as ssim_fn

# 读 50k_v2 的 skel 分片
per = 5056
def std_latent(idx):
    s = idx // per; o = idx % per
    p = "data/50k_v2_glyph15k/shards_std/shard_%05d.npz" % s
    if not os.path.exists(p):
        return None
    d = np.load(p)
    if o >= len(d["latents"]):
        return None
    return torch.from_numpy(d["latents"][o:o+1]).float()

rows = []
missing = 0
for i, r in df.iterrows():
    oid = r.get("old_50k_id")
    if pd.isna(oid):
        missing += 1; continue
    lat = std_latent(int(oid))
    if lat is None:
        missing += 1; continue
    pid = int(r["slot_id"]) if "slot_id" in df.columns else None
    if pid is None or pid < 0 or pid >= 23:
        missing += 1; continue
    e = emb[pid:pid+1]
    with torch.no_grad():
        pred = sk(lat, e)
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
    rows.append((i, r["character"], r["calligrapher"], pid, lat, pred))

print()
print("成功取样:", len(rows), " 缺失:", missing)
if not rows:
    sys.exit(0)

# 比较"骨架 latent 距真实书法骨架的距离"
# GT: 用 csv 的 std_path 反查 —— 但 strict 的 std_path 指向 data/50k/std (真实书法骨架)
# 因为我们要评的是"骨架几何", 用 latent 域的 MSE 太抽象, 直接比 CSR 指标
diffs, base_norms, pred_norms = [], [], []
for i, ch, cal, pid, lat, pred in rows:
    d = (pred - lat)
    diffs.append(d.abs().mean().item())
    base_norms.append(lat.abs().mean().item())
    pred_norms.append(pred.abs().mean().item())

print()
print("SkelNet 形变统计 (strict 84, top10 表):")
print("  |S_std|_mean  = %.4f" % np.mean(base_norms))
print("  |S_pred|_mean = %.4f" % np.mean(pred_norms))
print("  |S_pred-S_std|_mean = %.4f (%.1f%% of input scale)" % (
    np.mean(diffs), 100 * np.mean(diffs) / max(1e-9, np.mean(base_norms))))

# 按书家分组看形变幅度
sub = pd.DataFrame({"char": [r[1] for r in rows], "cal": [r[2] for r in rows],
                    "pid": [r[3] for r in rows], "diff": diffs})
print()
print("  按风格槽位形变幅度:")
print(sub.groupby("pid")["diff"].agg(["count", "mean"]).to_string())
