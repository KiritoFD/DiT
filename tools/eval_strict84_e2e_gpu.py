#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_strict84_e2e_gpu.py — 在 GPU 上以正统两阶段流水线快速评估 Strict84 端到端真实成绩"""
import os, sys, json, csv, re, time, argparse, math
import numpy as np
import torch as th
from PIL import Image

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)

dev = th.device("cuda" if th.cuda.is_available() else "cpu")

from src.eval import model_io
from src.eval.in_mem_eval import _get_vae
from src.eval.metrics import ssim_torch
from src.eval.inference import sample_latents, build_diffusion
from src.utils.callig_script_map import map_callig_script
import torchvision.transforms as T


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="exp/v35_union/20261002-003600-v35-union-1step/checkpoints/0010000.pt")
    ap.add_argument("--gen-steps", type=int, default=25)
    ap.add_argument("--bak-steps", type=int, default=50)
    ap.add_argument("--batch", type=int, default=28)
    ap.add_argument("--out-dir", default="exp/v35_union/eval_strict84_e2e_final")
    return ap.parse_args()


def main():
    a = parse_args()
    print(f"[1] Loading ckpt: {a.ckpt} on {dev}...")
    ck = th.load(a.ckpt, map_location="cpu")
    gen_sd = ck["gen_ema"]
    gen_ckpt = ck["gen_ckpt"]
    bak_ckpt = ck["bak_ckpt"]

    gen, _ = model_io.load_model_from_ckpt(gen_ckpt, device=dev, use_ema=True)
    bak, _ = model_io.load_model_from_ckpt(bak_ckpt, device=dev, use_ema=True)

    gen.load_state_dict({(k[10:] if k.startswith("_orig_mod.") else k): v for k, v in gen_sd.items()})
    gen.eval()
    bak.eval()

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
    print(f"[2] 载入 Strict84 测试集: {len(rows)} 样本")

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

    diff_skel = build_diffusion(a.gen_steps, "flow")
    diff_img = build_diffusion(a.bak_steps, "flow")

    g_gen = th.Generator(device=dev).manual_seed(0)
    noise_skel = th.randn(len(rows), 4, 32, 32, generator=g_gen, device=dev)
    noise_img = th.randn(len(rows), 4, 32, 32, generator=g_gen, device=dev)

    print("\n[3] 运行正统两阶段端到端推理:")
    t0 = time.time()
    with th.no_grad():
        print(f"  3.1 Stage 1 生成预测骨架 ({a.gen_steps} 步)...")
        g_pred = sample_latents(gen, diff_skel, noise_skel, conds, cfg_scale=1.0, batch=a.batch, device=dev, skel=g_stds)
        
        print(f"  3.2 Stage 2 成画渲染 ({a.bak_steps} 步, 骨架全程稳定保持)...")
        x_pred = sample_latents(bak, diff_img, noise_img, conds, cfg_scale=1.0, batch=a.batch, device=dev, skel=g_pred)
    dt_sample = time.time() - t0
    print(f"      GPU 采样全部完成! 总耗时: {dt_sample:.2f}s")

    print("\n[4] VAE 解码与指标计算...")
    vae = _get_vae(dev, "data/pretrained/pretrained_models/sd-vae-ft-ema")
    decs = []
    with th.no_grad():
        for s in range(0, len(rows), 28):
            _lat = x_pred[s:s+28].to(dev) / 0.18215
            _dec = (vae.decode(_lat).sample.clamp(-1, 1) + 1) / 2
            decs.append(_dec.cpu())
    dec = th.cat(decs, dim=0)
    gts_norm = (gts.cpu() + 1) / 2

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
    print(f"=== Strict84 真实真实端到端 ({os.path.basename(a.ckpt)}) 评测成绩 ===")
    print("="*60)
    print(f"样本数 n           : {len(rows)}")
    print(f"端到端 SSIM (均值) : {mean_ssim:.4f}")
    print(f"端到端 SSIM (中位) : {med_ssim:.4f}")
    print(f"均方误差 MSE       : {mean_mse:.4f}")
    print(f"墨迹 SSIM          : {mean_ink:.4f}")
    print("="*60)

    from collections import defaultdict
    by_cal = defaultdict(list)
    for r, s in zip(rows, ssim_vals):
        by_cal[r["calligrapher"]].append(s)

    print("\n【各书家 Strict84 端到端平均 SSIM 明细】:")
    for cal, vals in sorted(by_cal.items()):
        print(f"  {cal:8s} (n={len(vals):2d}): SSIM = {np.mean(vals):.4f} (med={np.median(vals):.4f})")

    os.makedirs(a.out_dir, exist_ok=True)
    json.dump({
        "ckpt": a.ckpt,
        "n": len(rows),
        "ssim_mean": mean_ssim,
        "ssim_median": med_ssim,
        "mse": mean_mse,
        "ink_ssim": mean_ink,
        "by_calligrapher": {k: float(np.mean(v)) for k, v in by_cal.items()}
    }, open(os.path.join(a.out_dir, "metrics.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    # 渲染海报 (拼图对比)
    # 每行 7 组样本，每组并排两格: [生成图 | 真实真迹]
    from torchvision.utils import save_image
    # 构造并排图像 (N, 2, C, H, W) -> (N*2, C, H, W)
    pairs = th.stack([dec.cpu(), gts_norm.cpu()], dim=1).reshape(-1, 3, 256, 256)
    poster_p = os.path.join(a.out_dir, "strict84_real_poster.png")
    save_image(pairs, poster_p, nrow=14, padding=2, normalize=False)
    print(f"✓ 海报已渲染保存: {poster_p}")
    print(f"✓ 指标已存盘: {a.out_dir}/metrics.json")


if __name__ == "__main__":
    main()
