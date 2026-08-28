# -*- coding: utf-8 -*-
"""
src.eval.inference — 统一推理核心 (DDPM 与 Flow-Matching 通用).

设计原则:
  * 一个核心: 模型采样 (bf16) → VAE decode (fp32) → PNG 落盘 → CPU 指标。
  * 其余 eval 模块 (in_process_eval / in_process_ctrl_eval / auto_eval_*)
    只是「配置 + 调用」的薄壳, 不再各自实现采样循环 (避免复制-漂移)。
  * 时间步约定:
      - Flow:   t ∈ [0,1) 连续, 模型输入 t*1000; Euler ODE 采样 (t: 1→0)。
      - DDPM:   t ∈ {0..T-1} 整数; DDIM 采样。
    采样统一走 ``diffusion.ddim_sample_loop(..., clip_denoised=False)``:
    对 flow 是 Euler (velocity 不可 clip), 对 ddpm 是 DDIM; 由 diffusion 对象
    内部决定, 调用方不分支。
"""
import os
import time
import json
import csv
import re

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

from src.loss import create_diffusion_or_flow


# ── VAE: 进程内单例 (GPU 推理进程复用) ──────────────────────────────────────
_eval_vae = None
_eval_vae_ref = None


def load_eval_vae(device, vae_path=None):
    """Lazily load the VAE once per process (modules that are eval shells share it)."""
    global _eval_vae, _eval_vae_ref
    if _eval_vae is not None and _eval_vae_ref is device:
        return _eval_vae
    from diffusers.models import AutoencoderKL
    if vae_path and os.path.exists(vae_path):
        _eval_vae = AutoencoderKL.from_pretrained(vae_path).to(device).eval()
    else:
        _eval_vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device).eval()
    for p in _eval_vae.parameters():
        p.requires_grad_(False)
    _eval_vae_ref = device
    return _eval_vae


# ── 扩散对象 ─────────────────────────────────────────────────────────────────
def build_diffusion(steps, diffusion_type="ddpm", flow_kwargs=None):
    """Build the diffusion sampler for eval. steps: int (ODE/DDIM step count).

    ``flow_kwargs`` (dict) 透传给 FlowMatching —— 例如
    ``{"t_sampler": "logit_normal", "sampler": "heun", "shift": 1.0}``。
    推理侧必须与训练侧使用同一份 flow 配置，否则会静默地用另一套
    时间步分布 / 求解器去评估模型。
    """
    return create_diffusion_or_flow(str(steps), diffusion_type=diffusion_type,
                                    **(flow_kwargs or {}))


# ── 采样 (GPU, bf16) → latents on CPU ───────────────────────────────────────
@torch.no_grad()
def sample_latents(model, diffusion, noise, conds, cfg_scale, batch, device,
                   skel=None, seed=0):
    """Run diffusion sampling (DDIM for ddpm / Euler for flow) → CPU latents.

    model       : callable(x, t, y_callig, y_char, cfg_scale=..., cond=...) —
                  i.e. ``model.forward_with_cfg`` (CFG handled at the model level).
    noise       : (N, C, H, W) fixed noise (CPU).
    conds       : list of (callig_id, glyph_id) tuples, length N.
    cfg_scale   : classifier-free guidance scale (0 = no CFG).
    skel        : optional structural condition (ControlNet path):
                  (N,4,32,32) VAE latent (新) 或 (N,1,256,256) PNG (旧).
    Returns (N, C, H, W) float32 latents on CPU.
    """
    n = noise.shape[0]
    lc, ls = noise.shape[1], noise.shape[2]
    all_latents = torch.zeros(n, lc, ls, ls, dtype=torch.float32)
    torch.manual_seed(seed)  # deterministic per call; noise itself is fixed anyway
    # CFG 在模型层处理 (forward_with_cfg)。不能把 cfg_scale 塞进 model_kwargs:
    # sampler 只做 model(x, t, **kwargs) 转发, plain forward 收到 cfg_scale 会
    # 直接 TypeError (base 通道崩溃), ctrl 通道则静默吞掉 → CFG 从未生效。
    if cfg_scale and cfg_scale > 0:
        def model_fn(x, t, **kw):
            return model.forward_with_cfg(x, t, cfg_scale=cfg_scale, **kw)
    else:
        model_fn = model
    for i in range(0, n, batch):
        j = min(i + batch, n)
        z = noise[i:j].to(device)
        yc = torch.tensor([c[0] for c in conds[i:j]], device=device, dtype=torch.long)
        yh = torch.tensor([c[1] for c in conds[i:j]], device=device, dtype=torch.long)
        mk = dict(y_callig=yc, y_char=yh)
        if skel is not None:
            mk["cond"] = skel[i:j].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            samples = diffusion.ddim_sample_loop(
                model_fn, z.shape, z,
                clip_denoised=False, model_kwargs=mk, device=device)
        all_latents[i:j] = samples.float().cpu()
        del z, samples
        torch.cuda.empty_cache()
    return all_latents


# ── VAE decode (fp32) → PNG 落盘 ────────────────────────────────────────────
@torch.no_grad()
def decode_and_save(vae, latents, scaling_factor, out_dir, tag, conds=None,
                    gts=None, vae_batch=16, skels=None):
    """Decode latents (fp32, force_upcast) → save {tag}{i}.png [+gt{i}.png, skel{i}.png].

    latents      : (N, C, H, W) CPU float32.
    scaling_factor: VAE latent scaling (e.g. 0.18215).
    out_dir      : directory to write PNGs into (created).
    tag          : image prefix (e.g. 'ctrl' / 'base' / 'sample').
    conds/gts/skels: optional metadata / GT images (N,3,H,W) [-1,1] / skels.
    Returns number of saved images.
    """
    os.makedirs(out_dir, exist_ok=True)
    n = latents.shape[0]
    n_saved = 0
    for i in range(0, n, vae_batch):
        j = min(i + vae_batch, n)
        lat = latents[i:j].to(latents.device if latents.is_cuda else next(vae.parameters()).device)
        decoded = vae.decode(lat / scaling_factor).sample  # fp32
        preds = decoded.float().cpu()
        for k in range(j - i):
            idx = i + k
            p = ((preds[k].clamp(-1, 1) + 1) / 2).clamp(0, 1)
            Image.fromarray((p.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(
                os.path.join(out_dir, f"{tag}{idx}.png"))
            if gts is not None:
                g = ((gts[i + k].clamp(-1, 1) + 1) / 2).clamp(0, 1)
                Image.fromarray((g.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(
                    os.path.join(out_dir, f"gt{idx}.png"))
            if skels is not None:
                # skel 可视化: latent (N,4,32,32) 先 VAE decode; PNG (N,1,H,W) 直接用
                if skels.ndim == 4 and skels.shape[1] == 4:
                    lat_sk = skels[i + k:i + k + 1].to(
                        next(vae.parameters()).device)
                    dec_sk = vae.decode(lat_sk / scaling_factor).sample.float().cpu()[0]
                    s = ((dec_sk.clamp(-1, 1) + 1) / 2).clamp(0, 1).permute(1, 2, 0).numpy()
                    Image.fromarray((s * 255).astype(np.uint8)).save(
                        os.path.join(out_dir, f"skel{idx}.png"))
                else:
                    s = skels[i + k, 0].numpy()
                    Image.fromarray((s * 255).astype(np.uint8)).save(
                        os.path.join(out_dir, f"skel{idx}.png"))
        n_saved += j - i
        del lat, decoded, preds
        torch.cuda.empty_cache()
    return n_saved


# ── CPU 指标 (从图片 PNG 计算, 与 eval_ctrl_metrics_daemon 约定一致) ────────
def _mse(pred, gt):
    return float(np.mean((pred - gt) ** 2)) * 4.0


def _ssim(pred, gt, win=11, data_range=1.0, sigma=1.5):
    from scipy.ndimage import correlate1d
    radius = win // 2
    x_k = np.arange(-radius, radius + 1, dtype=np.float64)
    k1d = np.exp(-(x_k ** 2) / (2 * sigma ** 2))
    k1d /= k1d.sum()
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssims = []
    for ch in range(pred.shape[2]):
        x = pred[:, :, ch].astype(np.float64)
        y = gt[:, :, ch].astype(np.float64)
        def _g(img):
            return correlate1d(correlate1d(img, k1d, axis=0, mode='reflect'),
                               k1d, axis=1, mode='reflect')
        mu_x = _g(x); mu_y = _g(y)
        mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
        sx2 = _g(x * x) - mu_x2
        sy2 = _g(y * y) - mu_y2
        sxy = _g(x * y) - mu_xy
        ssim_map = ((2 * mu_xy + c1) * (2 * sxy + c2)) / \
                   ((mu_x2 + mu_y2 + c1) * (sx2 + sy2 + c2))
        ssims.append(ssim_map.mean())
    return float(np.mean(ssims))


def _skel_iou(pred, gt, thresh=0.5):
    try:
        from skimage.morphology import skeletonize
    except ImportError:
        from scipy.ndimage import binary_erosion, generate_binary_structure
        def skeletonize(binary):
            skel = np.zeros_like(binary)
            img = binary.copy()
            struct = generate_binary_structure(2, 2)
            while img.any():
                eroded = binary_erosion(img, structure=struct)
                skel |= img & ~eroded
                img = eroded
            return skel
    # 兼容单张 (H,W,3) 与 batch (N,H,W,3) 两种输入
    if pred.ndim == 3:
        pred = pred[None]
        gt = gt[None]
    g1 = pred.mean(axis=3); g2 = gt.mean(axis=3)
    b1 = g1 < thresh; b2 = g2 < thresh
    inter_sum = union_sum = 0.0
    for k in range(pred.shape[0]):
        if not b1[k].any() and not b2[k].any():
            inter_sum += 1.0; union_sum += 1.0; continue
        if not b1[k].any() or not b2[k].any():
            union_sum += 1.0; continue
        s1 = skeletonize(b1[k]); s2 = skeletonize(b2[k])
        inter_sum += float((s1 & s2).sum()); union_sum += float((s1 | s2).sum())
    return inter_sum / union_sum if union_sum > 0 else 1.0


_lpips_fn = None
_lpips_loaded = False


def _get_lpips():
    global _lpips_fn, _lpips_loaded
    if _lpips_loaded:
        return _lpips_fn
    _lpips_loaded = True
    try:
        import lpips
        _lpips_fn = lpips.LPIPS(net='vgg', verbose=False)
        _lpips_fn.eval()
        for p in _lpips_fn.parameters():
            p.requires_grad_(False)
    except Exception:
        _lpips_fn = None
    return _lpips_fn


def compute_metrics(dec_dir, gt_dir, tag_prefix, n, use_lpips=True):
    """Compute MSE/SSIM/skel_iou (optional LPIPS) from PNG pairs on CPU.

    dec_dir : dir with {tag_prefix}{i}.png
    gt_dir  : dir with gt{i}.png (== dec_dir in the ctrl eval layout)
    Returns dict of scalar metrics.
    """
    lpips_fn = _get_lpips() if use_lpips else None
    mses, ssims, skels, lpips_ = [], [], [], []
    for i in range(n):
        p = os.path.join(dec_dir, f"{tag_prefix}{i}.png")
        g = os.path.join(gt_dir, f"gt{i}.png")
        if not (os.path.exists(p) and os.path.exists(g)):
            continue
        pred = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
        gt = np.asarray(Image.open(g).convert("RGB"), np.float32) / 255.0
        mses.append(_mse(pred, gt))
        ssims.append(_ssim(pred, gt))
        skels.append(_skel_iou(pred, gt))
        if lpips_fn is not None:
            import torch as _t
            pp = _t.from_numpy(pred.transpose(2, 0, 1)[None] * 2 - 1)
            gg = _t.from_numpy(gt.transpose(2, 0, 1)[None] * 2 - 1)
            with _t.no_grad():
                lpips_.append(float(lpips_fn(pp, gg).mean().item()))
    res = {"n": len(mses)}
    if mses:
        res["mse_mean"] = float(np.mean(mses))
        res["mse_std"] = float(np.std(mses))
        res["mse_q25"], res["mse_q50"], res["mse_q75"] = [float(q) for q in np.percentile(mses, [25, 50, 75])]
        res["ssim_mean"] = float(np.mean(ssims))
        res["ssim_std"] = float(np.std(ssims))
        res["skel_iou_mean"] = float(np.mean(skels))
        res["skel_iou_std"] = float(np.std(skels))
    if lpips_:
        res["lpips_mean"] = float(np.mean(lpips_))
    return res


# ── eval 条件缓存 (GT 图 + conds + skel + 固定 noise) ───────────────────────
def make_eval_cache(eval_csv, img_root, skel_root, image_size, n,
                    vae_downscale, latent_channels, scaling_factor,
                    skel_latent_shards_dir=None):
    """Pre-load N eval samples: GT images + conditions + skels + fixed noise (CPU).

    skel 条件: 优先从 skel_latent_shards_dir 加载 VAE latent (N,4,32,32);
    否则从 skel_root 读 PNG (N,1,256,256)。两者都存: skels_latent 供采样,
    skels (PNG 时) 供 skel{i}.png 可视化。
    """
    rows = list(csv.DictReader(open(eval_csv, encoding="utf-8")))
    if n > len(rows):
        n = len(rows)
    rows = rows[:n]
    transform = T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])
    latent_spatial = image_size // vae_downscale
    gts = torch.zeros(n, 3, image_size, image_size, dtype=torch.float32)
    conds = []
    skels = torch.zeros(n, 1, image_size, image_size, dtype=torch.float32)
    skels_latent = torch.zeros(
        n, latent_channels, latent_spatial, latent_spatial, dtype=torch.float32) \
        if skel_latent_shards_dir else None

    # 预建 skel latent shard 索引 (id -> (shard_path, offset))
    skel_id_to_shard = {}
    if skel_latent_shards_dir:
        import glob as _glob
        for sp in sorted(_glob.glob(os.path.join(skel_latent_shards_dir, "shard_*.npz"))):
            with np.load(sp) as d:
                for j, iid in enumerate(d["img_ids"]):
                    skel_id_to_shard[int(iid)] = (sp, j)

    for i, row in enumerate(rows):
        p = row["image_path"]
        if img_root and not os.path.isabs(p) and not p.startswith(img_root):
            p = os.path.join(img_root, p)
        gts[i] = transform(Image.open(p).convert("RGB"))
        conds.append((int(row["calligrapher_id"]),
                      int(row.get("glyph_id", row.get("character_id", 0)))))
        m = re.search(r"(\d+)\.png", p)
        img_id = int(m.group(1)) if m else None
        if img_id is not None and skels_latent is not None:
            if img_id in skel_id_to_shard:
                sp, j = skel_id_to_shard[img_id]
                with np.load(sp) as d:
                    skels_latent[i] = torch.from_numpy(np.array(d["latents"][j], copy=True)).float()
        elif img_id is not None and skel_root:
            sk = Image.open(os.path.join(skel_root, f"{img_id}.png")).convert("L")
            sk = sk.resize((image_size, image_size), Image.NEAREST)
            skels[i, 0] = torch.from_numpy(np.asarray(sk, np.float32) / 255.0)
    g = torch.Generator().manual_seed(0)
    noise = torch.randn(n, latent_channels, latent_spatial, latent_spatial, generator=g)
    return {"gts": gts, "conds": conds, "noise": noise, "skels": skels,
            "skels_latent": skels_latent,
            "n": n, "latent_channels": latent_channels,
            "latent_spatial": latent_spatial, "scaling_factor": scaling_factor,
            "img_root": img_root, "skel_root": skel_root, "image_size": image_size}


# ── 高层组合: pair eval (base vs ctrl) ──────────────────────────────────────
@torch.no_grad()
def run_pair_eval(model, vae, diffusion, cache, device, step, checkpoint_dir,
                  ddim_steps=50, cfg_scale=4.0, dit_batch=16, vae_batch=16,
                  with_skel=True, tag="ctrl"):
    """Sample (with optional skel) → decode → save PNGs under
    eval_samples_ctrl/stepXXXXXXX/{tag}/. GPU-only; metrics come from a CPU daemon.
    Returns (n_saved, elapsed).
    """
    t0 = time.time()
    n = cache["n"]
    lc, ls, sf = cache["latent_channels"], cache["latent_spatial"], cache["scaling_factor"]
    conds, gts_all, noise_all = cache["conds"], cache["gts"], cache["noise"]
    skels = cache.get("skels")
    skels_latent = cache.get("skels_latent")

    step_tag = f"step{int(step):07d}"
    out_dir = os.path.join(checkpoint_dir, "eval_samples_ctrl", step_tag, tag)
    os.makedirs(out_dir, exist_ok=True)

    # 采样条件: skel VAE latent 优先 (与训练一致), 否则 PNG (旧行为)
    skel_cond = skels_latent if skels_latent is not None else skels
    skel_arg = skel_cond if with_skel else None
    latents = sample_latents(model, diffusion, noise_all, conds, cfg_scale,
                             dit_batch, device, skel=skel_arg, seed=0)
    n_saved = decode_and_save(vae, latents, sf, out_dir, tag,
                              gts=gts_all, skels=skel_cond if with_skel else None,
                              vae_batch=vae_batch)
    elapsed = time.time() - t0
    with open(os.path.join(out_dir, "samples.json"), "w") as f:
        json.dump({"step": step, "n": n_saved, "cfg": cfg_scale,
                   "ddim_steps": ddim_steps, "tag": tag}, f, ensure_ascii=False)
    return n_saved, elapsed


def write_pending_metrics_marker(checkpoint_dir, step, n_base, n_ctrl, elapsed_base,
                                 elapsed_ctrl, ddim_steps, cfg_scale):
    """Write eval_pending_ctrl_{step}.json consumed by the CPU metrics daemon.

    Field ``step_tag`` is what eval_ctrl_metrics_daemon.process_pending reads
    to locate eval_samples_ctrl/{step_tag}/{base,ctrl}/.
    """
    step_tag = f"step{int(step):07d}"
    pending = {
        "step": step,
        "step_tag": step_tag,
        "n": n_ctrl,
        "nb": n_base,
        "elapsed_base": elapsed_base,
        "elapsed_ctrl": elapsed_ctrl,
        "ddim_steps": ddim_steps,
        "cfg_scale": cfg_scale,
    }
    pending_path = os.path.join(checkpoint_dir, f"eval_pending_ctrl_{int(step):07d}.json")
    with open(pending_path, "w") as f:
        json.dump(pending, f, indent=2)
    return pending_path