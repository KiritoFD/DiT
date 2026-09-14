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
from PIL import Image

from src.eval.inference import (build_diffusion, load_eval_vae, make_eval_cache,
                                sample_latents, sample_latents_self_cond,
                                _mse, _ssim)

# 模块级缓存: 跨 step 复用 (eval cache / VAE / diffusion / id_map)
_CACHES = {}
_VAE = None
_DIFF = None
_CMAP = None
_WHITE_LAT_CACHE = {}       # 白底 latent (aux_zero_white 时 decode 前加回)


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


def _poster_canny(img):
    """从 gen 图现算 canny 边缘列 (与老 make_seen_poster 同逻辑)。"""
    import cv2
    a = np.asarray(img, dtype=np.float32)
    gray = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    gx = cv2.filter2D(gray, -1, kx, borderType=cv2.BORDER_REFLECT)
    gy = cv2.filter2D(gray, -1, ky, borderType=cv2.BORDER_REFLECT)
    return Image.fromarray(((np.sqrt(gx ** 2 + gy ** 2)) > 150).astype(np.uint8) * 255).convert("RGB")


def _poster_skeleton(img):
    """从 gen 图现算骨架列 (与老 make_seen_poster 同逻辑)。"""
    from skimage.morphology import skeletonize
    a = np.asarray(img, dtype=np.float32)
    gray = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    bin_b = (gray > 127).astype(np.uint8)
    if gray.mean() > 127:
        bin_b = 1 - bin_b
    sk = skeletonize(bin_b.astype(bool)).astype(np.uint8) * 255
    return Image.fromarray(sk).convert("RGB")


def save_input_g(results_dir, set_name, skels_latent, vae, sf):
    """标准字输入 (g 条件) 落盘: eval_samples_ctrl/{set}_input_g/g{i}.png (幂等)。"""
    out_dir = os.path.join(results_dir, "eval_samples_ctrl", f"{set_name}_input_g")
    os.makedirs(out_dir, exist_ok=True)
    n = skels_latent.shape[0]
    if all(os.path.exists(os.path.join(out_dir, f"g{i}.png")) for i in range(n)):
        return out_dir
    with torch.autocast("cuda", dtype=torch.bfloat16):
        dec = vae.decode(skels_latent.to(next(vae.parameters()).device) / sf).sample
    dec = ((dec.float().clamp(-1, 1) + 1) / 2).cpu().numpy().transpose(0, 2, 3, 1)
    from PIL import Image as _Img
    for i in range(n):
        _Img.fromarray((dec[i] * 255).astype(np.uint8)).save(os.path.join(out_dir, f"g{i}.png"))
    return out_dir


def _steps_scan(results_dir, sub):
    base = os.path.join(results_dir, "eval_samples_ctrl")
    steps = []
    for d in sorted(glob.glob(os.path.join(base, "step*"))):
        n = 0
        while os.path.exists(os.path.join(d, sub, f"g{n}.png")):
            n += 1
        if n:
            steps.append((int(re.search(r"step(\d+)", os.path.basename(d)).group(1)),
                          os.path.join(d, sub), n))
    return steps


def render_poster(results_dir, set_name, out=None, cell=224, gap=6):
    """自动 poster (每 set 两张):

    主图 posters/{set}_poster.png:
      第 1 行 = 输入标准字 (eval_samples_ctrl/{set}_input_g/g{i}.png)
      中间每行 = 一个 ckpt 的生成结果 (时间升序, 行标签带 ssim)
      最后 1 行 = GT (最新 step 的 gt{i}.png)
    结构图 posters/{set}_struct.png (另放, 不挤主图):
      每 ckpt 一行, 每样本 gen | canny | skel 三列。
    均为全量重画并覆盖 → 永远是所有 step 的最新版。纯 CPU, 秒级。
    """
    from PIL import Image as _Img, ImageDraw
    sub = "g" if set_name in ("seen", "g") else set_name
    steps = _steps_scan(results_dir, sub)
    if not steps:
        return None
    n_max = max(s[2] for s in steps)
    cell = max(64, min(224, 1280 // max(n_max, 1)))
    input_dir = os.path.join(results_dir, "eval_samples_ctrl", f"{set_name}_input_g")
    has_input = os.path.isdir(input_dir) and bool(glob.glob(os.path.join(input_dir, "g*.png")))

    def _cell(path, bg):
        if path and os.path.exists(path):
            return _Img.open(path).convert("RGB").resize((cell, cell), _Img.LANCZOS)
        return _Img.new("RGB", (cell, cell), bg)

    font = _load_font(r"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", max(14, cell // 8))
    font_big = _load_font(r"/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", max(28, cell // 3))
    label_h = max(40, cell // 4)
    hdr_h = 56

    # ── 主图: input 行 + 每 step gen 行 + GT 行 ──────────────────────────
    W = cell * n_max + gap * 2
    n_rows = len(steps) + 1 + (1 if has_input else 0)
    H = hdr_h + gap + n_rows * (label_h + cell + gap) + gap + 30
    canvas = _Img.new("RGB", (W, H), (15, 17, 22))
    draw = ImageDraw.Draw(canvas)
    y = gap
    draw.rectangle([0, y, W, y + hdr_h], fill=(0, 0, 0))
    draw.text((gap, y + 12), f"{set_name} (n={n_max}) — input g / per-ckpt gen / GT",
              font=font, fill=(255, 200, 120))
    y += hdr_h + gap
    if has_input:
        draw.text((gap, y + 8), "input (标准字 g)", font=font_big, fill=(120, 220, 255))
        y += label_h
        for i in range(n_max):
            canvas.paste(_cell(os.path.join(input_dir, f"g{i}.png"), (30, 40, 60)),
                         (gap + i * cell, y))
        y += cell + gap
    for step, d, n in steps:
        draw.text((gap, y + 8), f"step {step}  {_step_ssim_txt(results_dir, step, set_name)}",
                  font=font_big, fill=(160, 200, 255))
        y += label_h
        for i in range(n):
            canvas.paste(_cell(os.path.join(d, f"g{i}.png"), (40, 40, 40)), (gap + i * cell, y))
        y += cell + gap
    draw.text((gap, y + 8), "GT", font=font_big, fill=(255, 160, 160))
    y += label_h
    gt_dir = steps[-1][1]
    for i in range(n_max):
        canvas.paste(_cell(os.path.join(gt_dir, f"gt{i}.png"), (50, 50, 50)), (gap + i * cell, y))
    out = out or os.path.join(results_dir, "posters", f"{set_name}_poster.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    canvas.save(out)

    # ── 结构图: 每 step 一行, gen|canny|skel ─────────────────────────────
    W2 = cell * n_max * 3 + gap * 2
    H2 = hdr_h + gap + len(steps) * (label_h + cell + gap) + gap + 30
    canvas2 = _Img.new("RGB", (W2, H2), (15, 17, 22))
    draw2 = ImageDraw.Draw(canvas2)
    y = gap
    draw2.rectangle([0, y, W2, y + hdr_h], fill=(0, 0, 0))
    draw2.text((gap, y + 12), f"{set_name} structure — gen | canny | skel (per ckpt)",
               font=font, fill=(255, 200, 120))
    y += hdr_h + gap
    for step, d, n in steps:
        draw2.text((gap, y + 8), f"step {step}  {_step_ssim_txt(results_dir, step, set_name)}",
                   font=font_big, fill=(160, 200, 255))
        y += label_h
        for i in range(n):
            x = gap + i * 3 * cell
            gen = _cell(os.path.join(d, f"g{i}.png"), (40, 40, 40))
            canvas2.paste(gen, (x, y))
            canvas2.paste(_poster_canny(gen), (x + cell, y))
            canvas2.paste(_poster_skeleton(gen), (x + 2 * cell, y))
        y += cell + gap
    out2 = os.path.join(results_dir, "posters", f"{set_name}_struct.png")
    canvas2.save(out2)
    return out


def _load_font(fp, size):
    from PIL import ImageFont
    try:
        return ImageFont.truetype(fp, size)
    except Exception:
        return ImageFont.load_default()


def _step_ssim_txt(results_dir, step, set_name):
    sum_path = os.path.join(results_dir, "eval_stdskel_summary.csv")
    if os.path.exists(sum_path):
        for r in csv.DictReader(open(sum_path, encoding="utf-8")):
            if int(r["step"]) == step and r["set"] == set_name:
                return f"ssim={float(r['ssim_mean']):.4f}"
    return ""


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
                # 白底归零 (aux_zero_white): 训练目标已减去白底 latent,
                # decode 前必须**加回**, 否则整幅图偏色。
                if bool(getattr(args, "aux_zero_white", False)):
                    _wl = _WHITE_LAT_CACHE.get("w")
                    if _wl is None:
                        _p = "data/white_latent.npy"
                        if os.path.exists(_p):
                            _wl = torch.from_numpy(np.load(_p)).float().to(_lat.device)
                            _WHITE_LAT_CACHE["w"] = _wl
                    if _wl is not None:
                        _lat = _lat + _wl[None]
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
                # 标准字输入 (g 条件) 落盘 (poster 第 1 行, 跨 step 复用, 幂等)
                if cache.get("skels_latent") is not None:
                    try:
                        save_input_g(results_dir, name, cache["skels_latent"], vae, sf)
                    except Exception as _se:
                        logger(f"[in-mem-eval] input-g save failed: {_se!r}")

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
            # 自动 poster: 全量重画该 set 所有 step (秒级, 覆盖旧文件)
            try:
                _p = render_poster(results_dir, name)
                if _p:
                    logger(f"[in-mem-eval] poster updated: {_p}")
            except Exception as _pe:
                logger(f"[in-mem-eval] poster render failed: {_pe!r}")
    finally:
        f_sum.close()
        f_raw.close()
        torch.cuda.empty_cache()
    return out
