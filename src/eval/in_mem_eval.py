# -*- coding: utf-8 -*-
"""
in_mem_eval.py — 真·in-mem eval: 训练进程内, 暂停 stepping, 同卡采样+decode+指标
一次算完, 无 daemon、无 PNG 落盘、无 ckpt 重载。

用法 (config):
    "in_mem_eval": true,
    "in_mem_eval_sets": "seen:assets/eval_seen_v10.csv:10,strict:assets/eval_fame3_strict_clean_v9.csv:50",
    "eval_self_cond": true,          # noise400k 线的两遍自条件采样
    "eval_blend_alpha": 0.5,
    # 复用已有: eval_cfg, eval_steps, gpu_eval_img_root, gpu_eval_every, skel_latent_shards_dir

流程 (train.py 在 ckpt 保存点同步调用):
    ema_model (常驻 GPU, eval 模式) → make_eval_cache (模块级缓存, 跨 step 复用)
    → sample_latents[_self_cond] (bf16, dit_batch=16) → VAE decode in-mem (bf16)
    → PNG 落盘 (eval_samples_ctrl/step{step}/{seen|strict}/, g{i}.png + gt{i}.png)
    → _ssim/_mse → 追加 eval_stdskel_summary.csv / eval_stdskel_batch.csv
    (与 tools/eval/batch_eval.py 完全同格式 → registry / 历史曲线无缝兼容)

显存: 训练 17G + eval 模型常驻激活 + VAE ~0.6G + dit16/vae16 峰值 ~2G ≈ 19.5G < 24G。
"""
import csv
import glob
import os
import re
import time

import numpy as np
import torch

from src.eval.inference import (build_diffusion, load_eval_vae, make_eval_cache,
                                sample_latents, sample_latents_self_cond,
                                _mse, _ssim)

# 模块级缓存: 跨 step 复用 (eval cache / VAE / diffusion / id_map)
_CACHES = {}
_VAE = None
_DIFF = None
_CMAP = None


def _get_vae(device, vae_path="data/pretrained/sd-vae-ft-ema"):
    global _VAE
    if _VAE is None:
        _VAE = load_eval_vae(device, vae_path)
    return _VAE


def _get_callig_map(path):
    global _CMAP
    if _CMAP is None and path and os.path.exists(path):
        from src.utils.callig_map import load_callig_id_map
        _CMAP, _ = load_callig_id_map(path)
    return _CMAP


def _get_cache(csv_path, n, img_root, shards, args):
    """eval cache 跨 step 复用 (同一 set 每 2500 步重算一次无意义)。"""
    ck = (csv_path, n)
    if ck not in _CACHES:
        sf = float(getattr(args, "vae_scaling_factor", 0.18215))
        _CACHES[ck] = make_eval_cache(
            csv_path, img_root, None, 256, n, 8,
            int(getattr(args, "latent_channels", 4)), sf,
            skel_latent_shards_dir=shards,
            callig_id_map=_get_callig_map(getattr(args, "callig_id_map", None)))
    return _CACHES[ck]


@torch.no_grad()
def run_in_mem_eval(model, args, step, device, results_dir, sets=None,
                    logger=print):
    """采样→decode→指标一次算完, 返回 {set: ssim_mean}。

    model: ema_model (GPU, eval 模式, 有 forward_with_cfg)。
    sets: [(name, csv_path, n), ...] 默认 seen:10 + strict:50。
    """
    global _DIFF
    os.makedirs(results_dir, exist_ok=True)
    if sets is None:
        sets = []
        for spec in str(getattr(args, "in_mem_eval_sets", "") or "").split(","):
            if spec.strip():
                name, csvp, n = spec.strip().split(":")
                sets.append((name, csvp, int(n)))
    if not sets:
        return {}

    # csv image_path 已含完整相对路径 (data/imgs/...), img_root 置 None 避免重复拼接
    img_root = None
    shards = getattr(args, "skel_latent_shards_dir", "") or ""
    cfg_scale = float(getattr(args, "eval_cfg", 0.7))
    ddim_steps = int(getattr(args, "eval_steps", 50))
    dit_batch = int(getattr(args, "in_mem_eval_batch", 16))
    vae_batch = int(getattr(args, "in_mem_eval_vae_batch", 16))
    sf = float(getattr(args, "vae_scaling_factor", 0.18215))
    use_self_cond = bool(getattr(args, "eval_self_cond", False))
    blend_alpha = float(getattr(args, "eval_blend_alpha", 0.0))
    if _DIFF is None:
        _DIFF = build_diffusion(ddim_steps, str(getattr(args, "diffusion_type", "flow")))

    sum_path = os.path.join(results_dir, "eval_stdskel_summary.csv")
    raw_path = os.path.join(results_dir, "eval_stdskel_batch.csv")
    done = set()
    if os.path.exists(sum_path):
        for r in csv.DictReader(open(sum_path, encoding="utf-8")):
            done.add((int(r["step"]), r["set"]))
    new_sum = not os.path.exists(sum_path)
    new_raw = not os.path.exists(raw_path)
    f_sum = open(sum_path, "a", newline="", encoding="utf-8")
    f_raw = open(raw_path, "a", newline="", encoding="utf-8")
    w_sum = csv.writer(f_sum)
    w_raw = csv.writer(f_raw)
    if new_sum:
        w_sum.writerow(["exp", "step", "set", "n", "ssim_mean", "ssim_p10", "ssim_q1",
                        "ssim_med", "ssim_q3", "ssim_p90", "mse_mean", "lpips_mean"])
    if new_raw:
        w_raw.writerow(["exp", "step", "set", "idx", "img_id", "char", "script",
                        "mse", "ssim", "lpips"])
    exp = os.path.basename(results_dir.rstrip("/"))
    out = {}

    try:
        for name, csvp, n in sets:
            if (step, name) in done:
                continue
            cache = _get_cache(csvp, n, img_root, shards, args)
            n = cache["n"]
            t0 = time.time()
            if use_self_cond:
                lat = sample_latents_self_cond(
                    model, _DIFF, cache["noise"], cache["conds"],
                    cfg_scale, dit_batch, device,
                    skel=cache["skels_latent"], seed=0, blend_alpha=blend_alpha)
            else:
                lat = sample_latents(
                    model, _DIFF, cache["noise"], cache["conds"],
                    cfg_scale, dit_batch, device,
                    skel=cache["skels_latent"], seed=0)
            t_s = time.time() - t0

            vae = _get_vae(device)
            gts = (cache["gts"].to(device) + 1) / 2
            preds = torch.empty_like(gts)
            for i in range(0, n, vae_batch):
                j = min(i + vae_batch, n)
                _lat = lat[i:j].to(device)
                if _lat.shape[1] > 4:
                    _lat = _lat[:, :4]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    dec = vae.decode(_lat / sf).sample
                preds[i:j] = (dec.clamp(-1, 1) + 1) / 2
            pred_np = preds.cpu().numpy().transpose(0, 2, 3, 1)
            gt_np = gts.cpu().numpy().transpose(0, 2, 3, 1)

            # PNG 落盘 (与 batch_eval --save-samples 同路径): g{i}.png + gt{i}.png
            if bool(getattr(args, "in_mem_eval_save_samples", True)):
                from PIL import Image as _Img
                _sub = "g" if name in ("seen", "g") else name
                _sd = os.path.join(results_dir, "eval_samples_ctrl", f"step{int(step):07d}", _sub)
                os.makedirs(_sd, exist_ok=True)
                for i in range(n):
                    _Img.fromarray((pred_np[i] * 255).astype(np.uint8)).save(
                        os.path.join(_sd, f"g{i}.png"))
                    _Img.fromarray((gt_np[i] * 255).astype(np.uint8)).save(
                        os.path.join(_sd, f"gt{i}.png"))

            ssims, mses = [], []
            for i in range(n):
                mses.append(_mse(pred_np[i], gt_np[i]))
                ssims.append(_ssim(pred_np[i], gt_np[i]))
            ssim = np.array(ssims)
            mse = float(np.mean(mses))
            q10, q25, q50, q75, q90 = np.percentile(ssim, [10, 25, 50, 75, 90])
            w_sum.writerow([exp, step, name, n, f"{ssim.mean():.4f}",
                            f"{q10:.4f}", f"{q25:.4f}", f"{q50:.4f}",
                            f"{q75:.4f}", f"{q90:.4f}", f"{mse:.5f}", ""])
            for i in range(n):
                w_raw.writerow([exp, step, name, i, "", "", "",
                                f"{mses[i]:.5f}", f"{ssims[i]:.4f}", ""])
            f_sum.flush()
            f_raw.flush()
            out[name] = float(ssim.mean())
            logger(f"[in-mem-eval] step={step} set={name} n={n} "
                   f"ssim={ssim.mean():.4f} (med={q50:.4f}) mse={mse:.5f} "
                   f"sample={t_s:.0f}s total={time.time()-t0:.0f}s")
    finally:
        f_sum.close()
        f_raw.close()
        torch.cuda.empty_cache()
    return out
