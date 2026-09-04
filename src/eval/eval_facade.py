# -*- coding: utf-8 -*-
"""
eval_facade.py — 统一 in-process eval (base 预训练 + ctrl 后训练一致)。

设计原则 (2026-09-05 重构, 可靠性优先, 不追求极端 infra 速度):
  * 训练进程内 GPU 采样 (dit_batch 控制显存) → decode → **落盘 PNG** (可靠, 可复查)
  * **同进程**计算指标 (MSE/SSIM/skel_iou/LPIPS, CPU numpy/torch, 不走 daemon)
  * 写 eval_auto_{prefix}{step}.json (与旧 daemon 输出同结构, 兼容训练侧早停)
  * base (无 skel) 与 ctrl (base+skel 双采样) 统一入口; device 可选 cuda/cpu
    (CPU 模式慢, 但可用; GPU 空闲时选 cuda)

输出布局 (与旧 eval_samples_ctrl 兼容, 便于复用网格/样本脚本):
  {out_dir}/eval_samples_ctrl/stepNNNNNNN/{ctrl|base}/{ctrl|base}N.png + gtN.png (+skelN.png)
  {out_dir}/eval_auto_ctrl_{step}.json   (ctrl 阶段: ctrl+base 双指标)
  {out_dir}/eval_auto_{step}.json        (base 阶段: 仅 base 指标)

一致性: train.py / train_controlnet.py 都调 eval_run() 这一个入口,
参数 from config (eval_csv / eval_img_root / eval_skel_root / ... / eval_device)。
"""
import os
import time
import json
import datetime

# lazy imports (避免 import 时缺依赖)


def _log(msg):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [eval] {msg}", flush=True)


def build_eval_cache(eval_csv, img_root, skel_root, image_size=256, n=100,
                     vae_downscale=8, latent_channels=4, scaling_factor=0.18215,
                     skel_latent_shards_dir=None):
    """构建 eval cache (CPU 常驻: GT 图 + 条件 + skel latent + 固定 noise)。"""
    from .inference import make_eval_cache
    return make_eval_cache(
        eval_csv, img_root, skel_root, image_size, n,
        vae_downscale, latent_channels, scaling_factor,
        skel_latent_shards_dir=skel_latent_shards_dir)


def _load_vae(device, vae_path="pretrained_models/sd-vae-ft-ema"):
    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(vae_path).to(device).eval()
    for p in vae.parameters():
        p.requires_grad = False
    return vae


def _compute_and_save(vae, latents, cache, out_dir, tag, step_tag, device,
                      vae_batch=16, use_lpips=True, with_skel=False, skels_latent=None):
    """decode latents → 落盘 PNG (tag{i}.png + gt{i}.png [+skel{i}.png]) → 内存算指标。"""
    from .inference import decode_and_save, compute_metrics
    from PIL import Image
    import torch

    n = cache["n"]
    sf = cache["scaling_factor"]
    # decode + 落盘
    n_saved = decode_and_save(
        vae, latents, sf, out_dir, tag,
        gts=cache["gts"], skels=(skels_latent if with_skel else None),
        vae_batch=vae_batch)
    # 同进程算指标 (CPU numpy)
    metrics = compute_metrics(out_dir, out_dir, tag, n, use_lpips=use_lpips)
    return n_saved, metrics


def eval_run(model, diffusion, cache, device, step, out_dir,
             ddim_steps=50, cfg_scale=0.7, dit_batch=8, vae_batch=16,
             with_skel=True, vae_path="pretrained_models/sd-vae-ft-ema",
             use_lpips=True, n_samples=None):
    """统一 in-process eval: 采样(GPU/CPU) → 落盘 → 同进程指标 → eval_auto_*.

    base 预训练: with_skel=False → 只采样 base 一份 (tag="base", eval_auto_{step}.json)
    ctrl 后训练: with_skel=True → 采样 ctrl(带 skel) + base(无 skel) 两份
                 (eval_auto_ctrl_{step}.json, 含 ctrl/base/delta)

    返回 eval_auto json 路径。
    """
    t0 = time.time()
    from .inference import sample_latents

    vae = _load_vae(device, vae_path)
    step_tag = f"step{int(step):07d}"
    base_dir = os.path.join(out_dir, "eval_samples_ctrl", step_tag, "base")
    ctrl_dir = os.path.join(out_dir, "eval_samples_ctrl", step_tag, "ctrl")
    os.makedirs(base_dir, exist_ok=True)
    if with_skel:
        os.makedirs(ctrl_dir, exist_ok=True)

    noise = cache["noise"]
    conds = cache["conds"]
    skel_cond = cache.get("skels_latent")
    if n_samples is not None and n_samples < noise.shape[0]:
        noise = noise[:n_samples]
        conds = conds[:n_samples]
        if skel_cond is not None:
            skel_cond = skel_cond[:n_samples]

    res = {}
    # ---- ctrl (带 skel) ----
    if with_skel:
        lat = sample_latents(model, diffusion, noise, conds, cfg_scale,
                             dit_batch, device, skel=skel_cond, seed=0)
        n_s, m = _compute_and_save(vae, lat, cache, ctrl_dir, "ctrl", step_tag,
                                   device, vae_batch, use_lpips,
                                   with_skel=True, skels_latent=skel_cond)
        res["ctrl"] = m
        n_ctrl = n_s
        del lat
    # ---- base (无 skel) ----
    lat = sample_latents(model, diffusion, noise, conds, cfg_scale,
                         dit_batch, device, skel=None, seed=0)
    n_s, m = _compute_and_save(vae, lat, cache, base_dir, "base", step_tag,
                               device, vae_batch, use_lpips, with_skel=False)
    res["base"] = m
    n_base = n_s
    del lat

    # delta 指标 (ctrl 提升)
    if with_skel and "ctrl" in res and "base" in res:
        for k in ("mse", "ssim", "lpips"):
            ck, bk = f"{k}_mean", f"{k}_mean"
            if ck in res["ctrl"] and bk in res["base"]:
                res[f"delta_{k}"] = res["ctrl"][ck] - res["base"][bk]

    # 写 eval_auto_* (早停读这个)
    meta = {"step": step, "n": n_base, "cfg": cfg_scale, "ddim_steps": ddim_steps,
            "elapsed_s": round(time.time() - t0, 1)}
    if with_skel:
        out_json = os.path.join(out_dir, f"eval_auto_ctrl_{int(step)}.json")
        res.update(meta)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False)
        _log(f"[{step_tag}] ctrl(base) n={n_base} + ctrl(skel) n={res.get('ctrl', {}).get('n', 0)} "
             f"{time.time()-t0:.0f}s -> {os.path.basename(out_json)}")
        del vae
        return out_json
    else:
        out_json = os.path.join(out_dir, f"eval_auto_{int(step)}.json")
        base_res = {"n": n_base, "ssim": res["base"].get("ssim_mean"),
                    "lpips": res["base"].get("lpips_mean"),
                    "mse": res["base"].get("mse_mean"),
                    "ssim_std": res["base"].get("ssim_std"),
                    "lpips_std": res["base"].get("lpips_std"),
                    "mse_std": res["base"].get("mse_std"),
                    "skel_iou": res["base"].get("skel_iou_mean"),
                    "step": step, "cfg": cfg_scale, "ddim_steps": ddim_steps,
                    "elapsed_s": round(time.time() - t0, 1)}
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(base_res, f, ensure_ascii=False)
        _log(f"[{step_tag}] base n={n_base} {time.time()-t0:.0f}s -> {os.path.basename(out_json)}")
        del vae
        return out_json


def read_last_eval(out_dir, prefix="eval_auto_ctrl_", metric="ssim"):
    """读最近 completed eval (本进程写入), 供早停。返回 metric 值或 None。"""
    import glob
    files = sorted(glob.glob(os.path.join(out_dir, f"{prefix}*.json")))
    if not files:
        return None
    try:
        with open(files[-1], "r", encoding="utf-8") as f:
            d = json.load(f)
        if metric in d:
            return d[metric]
        if "ctrl" in d and metric in d["ctrl"]:
            return d["ctrl"].get(metric if metric != "ssim" else "ssim_mean")
        return d.get(metric + "_mean")
    except Exception:
        return None