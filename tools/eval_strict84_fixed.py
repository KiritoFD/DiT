#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_strict84_fixed.py — 使用修好的 JointSkel2Img (骨架轨迹级缓存, 条件稳定不变) 评测 Step 8,000 的真实 Strict84 E2E 表现"""
import os, sys, json, csv, re, time
import numpy as np
import torch as th
from PIL import Image

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)

th.set_num_threads(32)
dev = th.device("cpu")

from src.eval import model_io
from src.eval.in_mem_eval import _get_vae
from src.eval.metrics import ssim_torch
from src.eval.inference import sample_latents, build_diffusion
from src.model.joint_skel2img import JointSkel2Img
from src.utils.callig_script_map import map_callig_script
import torchvision.transforms as T

ckpt_path = "exp/v35_union/20261002-003600-v35-union-1step/checkpoints/0008000.pt"
print(f"[1] 载入 Checkpoint Step 8,000: {ckpt_path}", flush=True)
ck = th.load(ckpt_path, map_location="cpu")
gen_sd = ck["gen_ema"]
gen_ckpt = ck["gen_ckpt"]
bak_ckpt = ck["bak_ckpt"]

gen, ga = model_io.load_model_from_ckpt(gen_ckpt, device=dev, use_ema=True)
bak, ba = model_io.load_model_from_ckpt(bak_ckpt, device=dev, use_ema=True)

gen.load_state_dict({(k[10:] if k.startswith("_orig_mod.") else k): v for k, v in gen_sd.items()})
gen.eval()
bak.eval()

# 统一模型 (使用修好的 JointSkel2Img, 内部缓存 g_pred, 50 步扩散全程骨架稳定!)
wrapper = JointSkel2Img(gen, bak, gen_steps=25).to(dev).eval()
diff = build_diffusion(50, "flow")

# 官方 50k_v2 标准字骨架分片
shards_dir = "data/50k_v2_glyph15k/shards_std"
shard_cache = {}
id_to_shard = {}
import glob
for p in sorted(glob.glob(os.path.join(shards_dir, "shard_*.npz"))):
    with np.load(p) as z:
        for j, iid in enumerate(z["img_ids"]):
            id_to_shard[int(iid)] = (p, j)

def get_std_lat(iid):
    p, j = id_to_shard[iid]
    if p not in shard_cache:
        with np.load(p) as z:
            shard_cache[p] = np.array(z["latents"], copy=True)
    return shard_cache[p][j]

csv_path = "assets/eval_top10_strict_subset84.csv"
rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
print(f"[2] 载入 Strict84 测试集: {len(rows)} 样本", flush=True)

csmap = json.load(open("assets/callig_script_id_map_top10.json", encoding="utf-8"))

g_stds, conds, gts = [], [], []
tf = T.Compose([T.Resize((256, 256)), T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])

for r in rows:
    iid = int(re.search(r"(\d+)\.png", r["image_path"]).group(1))
    old_id = int(r.get("old_50k_id", iid))
    lat = get_std_lat(old_id if old_id in id_to_shard else iid)
    g_stds.append(lat)
    cid = map_callig_script(int(r["calligrapher_id"]), int(r["script_id"]), csmap)
    conds.append((cid, int(r.get("glyph_id", 0))))
    gts.append(tf(Image.open(r["image_path"]).convert("RGB")))

g_stds = th.from_numpy(np.stack(g_stds)).float().to(dev)
gts = th.stack(gts).to(dev)

g_gen = th.Generator().manual_seed(0)
noise = th.randn(len(rows), 4, 32, 32, generator=g_gen)

print("\n[3] 运行 JointSkel2Img 采样 (84 samples, 批次=14, 缓存骨架只采一次)...", flush=True)
t0 = time.time()
with th.no_grad():
    x_pred = sample_latents(wrapper, diff, noise, conds, cfg_scale=1.0, batch=14, device=dev, skel=g_stds)
t_sample = time.time() - t0
print(f"    采样完成! 耗时: {t_sample:.1f}s, x_pred 均值={x_pred.mean():.4f}, std={x_pred.std():.4f}", flush=True)

print("\n[4] 分批 VAE 解码与指标计算...", flush=True)
vae = _get_vae(dev, "data/pretrained/pretrained_models/sd-vae-ft-ema")
decs = []
with th.no_grad():
    for b_idx in range(0, len(rows), 14):
        _dec = (vae.decode(x_pred[b_idx:b_idx+14] / 0.18215).sample.clamp(-1, 1) + 1) / 2
        decs.append(_dec)
dec = th.cat(decs, dim=0)
gts_norm = (gts + 1) / 2

ssim_vals = ssim_torch(dec, gts_norm).cpu().numpy()
mse_vals = th.nn.functional.mse_loss(dec, gts_norm, reduction="none").mean([1,2,3]).cpu().numpy()

# 墨迹 SSIM
dec_gray = dec.mean(dim=1)
gt_gray = gts_norm.mean(dim=1)
dec_mask = (dec_gray < 0.6).float()
gt_mask = (gt_gray < 0.6).float()
ink_ssim_vals = ssim_torch(dec_mask.unsqueeze(1).repeat(1,3,1,1), gt_mask.unsqueeze(1).repeat(1,3,1,1)).cpu().numpy()

mean_ssim = float(np.mean(ssim_vals))
med_ssim = float(np.median(ssim_vals))
mean_mse = float(np.mean(mse_vals))
mean_ink = float(np.mean(ink_ssim_vals))

print("\n" + "="*60)
print("=== Strict84 真实端到端 (Step 8,000 Checkpoint) 评测成绩 ===")
print("="*60)
print(f"评估样本数 n       : {len(rows)}")
print(f"端到端 SSIM (均值) : {mean_ssim:.4f}")
print(f"端到端 SSIM (中位) : {med_ssim:.4f}")
print(f"均方误差 MSE       : {mean_mse:.4f}")
print(f"墨迹 SSIM          : {mean_ink:.4f}")
print("="*60)

# 书家分拆
from collections import defaultdict
by_cal = defaultdict(list)
for r, s in zip(rows, ssim_vals):
    by_cal[r["calligrapher"]].append(s)

print("\n【各书家 Strict84 端到端平均 SSIM 明细】:")
for cal, vals in sorted(by_cal.items()):
    print(f"  {cal:8s} (n={len(vals):2d}): SSIM = {np.mean(vals):.4f} (med={np.median(vals):.4f})")

out_dir = "exp/v35_union/eval_strict84_step8000"
os.makedirs(out_dir, exist_ok=True)
json.dump({
    "step": 8000,
    "n": len(rows),
    "ssim_mean": mean_ssim,
    "ssim_median": med_ssim,
    "mse": mean_mse,
    "ink_ssim": mean_ink,
    "by_calligrapher": {k: float(np.mean(v)) for k, v in by_cal.items()}
}, open(os.path.join(out_dir, "strict84_e2e_metrics.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print(f"\n✓ 评测结果已完整存盘: {out_dir}/strict84_e2e_metrics.json")
