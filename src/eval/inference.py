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
from src.utils.latent_dataset import extract_img_id

# img_id 提取失败只警告一次（避免 237 行刷屏）
_ID_WARNED = [False]


# ── VAE: 进程内单例 (GPU 推理进程复用) ──────────────────────────────────────
_eval_vae = None
_eval_vae_ref = None


def match_model_channels(noise, model, seed=0):
    """aux 目标通道 (in_channels>4) 时, 把噪声扩到模型输入通道数 (多余通道随机, 推理时丢弃).

    **确定性**: 用固定 seed 的生成器填充, 保证同一 ckpt 多次评测结果完全一致
    (否则随机 aux 经 attention 影响图像通道 -> ssim 抖动 ~0.02)。
    in_channels==4 时原样返回 (零开销, 兼容旧路径)。
    """
    import torch as _t
    tgt = int(getattr(model, "in_channels", noise.shape[1]))
    if noise.shape[1] == tgt:
        return noise
    if noise.shape[1] > tgt:
        return noise[:, :tgt]
    g = _t.Generator(device=noise.device).manual_seed(int(seed))
    extra = _t.randn(noise.shape[0], tgt - noise.shape[1], *noise.shape[2:],
                     dtype=noise.dtype, device=noise.device, generator=g)
    return _t.cat([noise, extra], dim=1)


def image_latent(lat):
    """取图像 latent (前 4 通道) —— aux 目标通道在解码/评测时丢弃。"""
    return lat[:, :4] if lat.shape[1] > 4 else lat


def load_eval_vae(device, vae_path=None):
    """Lazily load the VAE once per process (modules that are eval shells share it)."""
    global _eval_vae, _eval_vae_ref
    if _eval_vae is not None and _eval_vae_ref is device:
        return _eval_vae
    from diffusers.models import AutoencoderKL
    candidates = [
        vae_path,
        "data/pretrained/pretrained_models/sd-vae-ft-ema",
        "pretrained_models/sd-vae-ft-ema",
        "/root/Workspace/xy/DiT/data/pretrained/pretrained_models/sd-vae-ft-ema",
        "/root/Workspace/xy/DiT/pretrained_models/sd-vae-ft-ema",
    ]
    resolved = None
    for cand in candidates:
        if cand and os.path.exists(cand):
            resolved = cand
            break
    if resolved:
        _eval_vae = AutoencoderKL.from_pretrained(resolved).to(device).eval()
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
                   skel=None, seed=0, hier_conds=None):
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
    noise = match_model_channels(noise, model)   # aux 目标通道: 噪声扩到 in_channels
    lc, ls = noise.shape[1], noise.shape[2]
    all_latents = torch.zeros(n, lc, ls, ls, dtype=torch.float32)
    torch.manual_seed(seed)  # deterministic per call; noise itself is fixed anyway
    dev_type = device.type if isinstance(device, torch.device) else str(device)
    # CFG 在模型层处理 (forward_with_cfg)。不能把 cfg_scale 塞进 model_kwargs:
    # sampler 只做 model(x, t, **kwargs) 转发, plain forward 收到 cfg_scale 会
    # 直接 TypeError (base 通道崩溃), ctrl 通道则静默吞掉 → CFG 从未生效。
    if cfg_scale and cfg_scale > 0:
        # [2026-09-17] 双轴 CFG: 由**模型属性**驱动, 避免改 sample_latents 签名。
        #   cfg_glyph_scale 为 None -> 走经典 2 路 (向后兼容)
        #   非 None -> forward_with_cfg 内部委托给 forward_with_2axis_cfg
        #   ⚠ 仅在训练时 glyph_drop_prob>0 的 ckpt 上才有意义
        #     (g=0 必须是训练见过的条件, 否则内容轴未训练)
        _cg = getattr(model, "cfg_glyph_scale", None)
        _wi = float(getattr(model, "cfg_w_inter", 0.0) or 0.0)

        def model_fn(x, t, **kw):
            extra = {}
            if _cg is not None:
                extra["cfg_glyph"] = float(_cg)
                extra["w_inter"] = _wi
            return model.forward_with_cfg(x, t, cfg_scale=cfg_scale, **extra, **kw)
    else:
        model_fn = model
    for i in range(0, n, batch):
        j = min(i + batch, n)
        z = noise[i:j].to(device)
        yc = torch.tensor([c[0] for c in conds[i:j]], device=device, dtype=torch.long)
        yh = torch.tensor([c[1] for c in conds[i:j]], device=device, dtype=torch.long)
        mk = dict(y_callig=yc, y_char=yh)
        # ── S2 三层语义分解的三个 id（None = 模型走旧路径，零副作用）──────────
        if hier_conds is not None:
            _hraw, _hpair, _hscript = hier_conds
            mk['y_callig_raw'] = torch.tensor(_hraw[i:j], device=device, dtype=torch.long)
            mk['y_pair'] = torch.tensor(_hpair[i:j], device=device, dtype=torch.long)
            mk['y_script'] = torch.tensor(_hscript[i:j], device=device, dtype=torch.long)
        if skel is not None:
            # g 条件模型 (train.py use_glyph_cond/skel_as_glyph_cond 预训练) 走 'g' 键;
            # ControlNetDiT 包装走 'cond' 键 —— 按模型类型自动路由
            if getattr(model, "use_glyph_cond", False) or getattr(getattr(model, "main", None), "use_glyph_cond", False):
                mk["g"] = skel[i:j].to(device)
            else:
                mk["cond"] = skel[i:j].to(device)
        # bf16 autocast 仅 cuda (CPU 无 AVX512-BF16/AMX 时 bf16 走软件上转反而慢)
        if dev_type == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                samples = diffusion.ddim_sample_loop(
                    model_fn, z.shape, z,
                    clip_denoised=False, model_kwargs=mk, device=device)
        else:
            samples = diffusion.ddim_sample_loop(
                model_fn, z.shape, z,
                clip_denoised=False, model_kwargs=mk, device=device)
        all_latents[i:j] = samples.float().cpu()
        del z, samples
        if dev_type == "cuda":
            torch.cuda.empty_cache()
    return all_latents


# ── 两遍自条件采样 (Self-Conditioning) ───────────────────────────────────────
# 核心思路: 12ch 联合模型在推理时同时去噪 image/canny/skel 三组通道。
# 其中 skel 通道 (ch 8-11) 的预测实质是「模型认为这个书家会怎么写这个字的骨架」
# —— 即一个 **书家风格化骨架**，比通用印刷标准骨架 g 更接近 GT 骨架。
#
# 将 pass-1 预测的 skel 通道回灌为 pass-2 的条件 g，让模型在第二遍时拿到
# 一个自洽的、带书家个性的骨架条件，而非千篇一律的楷体印刷骨架。
#
# 开销: 约 1.4× (pass-1 用较少 ODE 步) 到 2× (pass-1 同等步数)。
# 不改训练，纯推理侧增益。

@torch.no_grad()
def sample_latents_self_cond(
    model, diffusion, noise, conds, cfg_scale, batch, device,
    skel=None, seed=0,
    hier_conds=None,
    # ── self-cond 参数 ──
    skel_ch_start=8, skel_ch_end=12,
    first_pass_steps=None,
    blend_alpha=0.0,
):
    """两遍自条件采样 (Self-Conditioning) for 12ch joint models.

    Pass 1: 用标准骨架 g 正常采样 → 得到 12ch 预测 (含 skel ch)
    Pass 2: 用 pass-1 预测的 skel 通道替换 g 重新采样 → 更优图像

    Args:
        skel_ch_start, skel_ch_end: 输出中骨架 latent 的通道范围
            (默认 8:12, 即 [img(0-3), canny(4-7), skel(8-11)])。
        first_pass_steps: pass-1 的 ODE 步数 (None = 与 pass-2 相同;
            设较小值如 20 可将总开销从 2× 降至 ~1.4×)。
        blend_alpha: pass-2 条件的混合系数:
            g_pass2 = (1 - α) · predicted_skel + α · g_original
            0.0 = 纯自条件; 0.5 = 各半; 1.0 = 无自条件 (退化为普通采样)。

    Returns:
        (N, C, H, W) float32 CPU latents (与 sample_latents 相同格式)。
    """
    # 非 12ch 模型或无 skel 条件 → 退化为普通采样
    _main = getattr(model, 'main', model)
    model_ch = int(getattr(_main, 'in_channels', 4))
    if model_ch <= 4 or skel is None or blend_alpha >= 1.0:
        return sample_latents(model, diffusion, noise, conds, cfg_scale,
                              batch, device, skel=skel, seed=seed,
                              hier_conds=hier_conds)

    # ── Pass 1: 用标准骨架采样，得到书家风格化骨架预测 ──
    if first_pass_steps is not None and first_pass_steps != diffusion.num_timesteps:
        diff1 = build_diffusion(
            first_pass_steps, diffusion_type='flow',
            flow_kwargs={
                'sampler': getattr(diffusion, 'sampler', 'heun'),
                'heun_batch': getattr(diffusion, 'heun_batch', True),
                'shift': getattr(diffusion, 'shift', 1.0),
            })
    else:
        diff1 = diffusion

    x0_pass1 = sample_latents(model, diff1, noise, conds, cfg_scale,
                              batch, device, skel=skel, seed=seed,
                              hier_conds=hier_conds)

    # 提取预测骨架 (pass-1 的 skel 通道)
    if x0_pass1.shape[1] <= skel_ch_start:
        # 模型输出通道不够 → 无 skel 可提取，退化为普通采样
        return x0_pass1

    predicted_skel = x0_pass1[:, skel_ch_start:skel_ch_end].clone()

    # 混合: 保留部分标准骨架信息 (α > 0 时对预测骨架做保守修正)
    if blend_alpha > 0:
        predicted_skel = (1.0 - blend_alpha) * predicted_skel + blend_alpha * skel

    # ── Pass 2: 用风格化骨架作为条件重新采样 ──
    x0_pass2 = sample_latents(model, diffusion, noise, conds, cfg_scale,
                              batch, device, skel=predicted_skel, seed=seed,
                              hier_conds=hier_conds)
    return x0_pass2


# ── 白底归零 (aux_zero_white) 的**唯一**加回入口 ──────────────────────────────
_WHITE_LAT = {}


def get_white_latent(device=None, path="data/white_latent.npy"):
    """白底图的 VAE latent (4,32,32)，已乘 scaling_factor（与训练目标同尺度）。"""
    z = _WHITE_LAT.get("w")
    if z is None:
        z = torch.from_numpy(np.load(path)).float()
        _WHITE_LAT["w"] = z
    return z if device is None else z.to(device)


def maybe_add_white(lat, zero_white=False):
    """decode 前把白底加回（当训练用 `tools/rebuild_latents_wz.py` 减过白底时）。

    ⚠ 为什么不内联在各处:
      2026-09-14 事故 —— 训练目标减了白底, 但 `aux_zero_white` 没在 train.py 注册,
      config 的值被静默丢弃, 于是**所有** decode 路径都没加回:
      VAE 把 latent 零向量解成灰黄棕色 (实测 RGB≈[129,110,89]), 更负处变黑,
      生成图整体发黄/发黑, poster 上看起来"全黑"。而训练本身完全正常。
      → 新增任何 decode 路径都必须走本函数, 不要各自实现。
    """
    if not zero_white:
        return lat
    try:
        w = get_white_latent(lat.device)
    except Exception:
        return lat
    if lat.dim() == 4 and lat.shape[1] != w.shape[0]:
        # aux 多通道 (如 8ch = canny4 + skel4 / 12ch = img4+canny4+skel4):
        # 白底 latent 固定 4ch, 每个 4ch 目标组都减过同一个白底,
        # 加回时按组重复到 C 通道, 否则 8ch + 4ch 维度不匹配 (2026-09-14 事故续)。
        c = lat.shape[1]
        if c % w.shape[0] != 0:
            raise ValueError(
                f"maybe_add_white: 通道数 {c} 不是 {w.shape[0]} 的整数倍, 无法按组加回白底")
        # ⚠ Tensor.repeat 的实参个数必须等于张量维数: (4,32,32) 要写 repeat(n,1,1),
        #   写成 repeat(n) 会直接报 "Number of dimensions of repeat dims can not be
        #   smaller than number of dimensions of tensor"。此前 eval 全线 FAILED 就是这个。
        w = w.repeat(c // w.shape[0], 1, 1)
    return lat + (w[None] if lat.dim() == 4 else w)


# ── VAE decode (fp32) → PNG 落盘 ────────────────────────────────────────────
@torch.no_grad()
def decode_and_save(vae, latents, scaling_factor, out_dir, tag, conds=None,
                    gts=None, vae_batch=16, skels=None, idx_offset=0,
                    zero_white=False):
    """Decode latents (fp32, force_upcast) → save {tag}{i}.png [+gt{i}.png, skel{i}.png].

    latents      : (N, C, H, W) CPU float32.
    scaling_factor: VAE latent scaling (e.g. 0.18215).
    out_dir      : directory to write PNGs into (created).
    tag          : image prefix (e.g. 'ctrl' / 'base' / 'sample').
    conds/gts/skels: optional metadata / GT images (N,3,H,W) [-1,1] / skels.
    idx_offset   : 全局下标偏移 (分段并行 eval 时 = 段起点, PNG 命名用全局下标).
    Returns number of saved images.
    """
    os.makedirs(out_dir, exist_ok=True)
    n = latents.shape[0]
    n_saved = 0
    vae_dev = next(vae.parameters()).device
    for i in range(0, n, vae_batch):
        j = min(i + vae_batch, n)
        lat = latents[i:j].to(vae_dev)
        if lat.shape[1] > 4:          # aux 目标通道: 解码只用图像 4 通道
            lat = lat[:, :4]
        lat = maybe_add_white(lat, zero_white)      # 白底归零: 减过的必须加回
        decoded = vae.decode(lat / scaling_factor).sample  # fp32
        preds = decoded.float().cpu()
        for k in range(j - i):
            idx = idx_offset + i + k
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
        if vae_dev.type == "cuda":
            torch.cuda.empty_cache()
    return n_saved


# ── CPU 指标 ────────────────────────────────────────────────────────────────
# ★ 2026-09-17: 实现已统一到 src/eval/metrics.py（仓库里曾有 14 份 SSIM / 8 份
#   skel_iou / 4 份 MSE 的复制粘贴，且 win 与输入约定不一致 -> 数字不可比）。
#   这里保留 `_mse` / `_ssim` / `_skel_iou` 三个名字做**向后兼容的再导出**，
#   所有既有调用点无需改动。口径与历史主评测路径完全一致（win=11 高斯窗）。
from src.eval.metrics import mse as _mse_impl, ssim as _ssim_impl, \
    skel_iou as _skel_iou_impl


def _mse(pred, gt):
    return _mse_impl(pred, gt)


def _ssim(pred, gt, win=11, data_range=1.0, sigma=1.5):
    return _ssim_impl(pred, gt, win=win, data_range=data_range, sigma=sigma)


def _skel_iou(pred, gt, thresh=0.5):
    return _skel_iou_impl(pred, gt, thresh=thresh)


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


def compute_metrics(dec_dir, gt_dir, tag_prefix, n, use_lpips=True, idx_range=None,
                    with_lists=False):
    """Compute MSE/SSIM/skel_iou (optional LPIPS) from PNG pairs on CPU.

    dec_dir : dir with {tag_prefix}{i}.png
    gt_dir  : dir with gt{i}.png (== dec_dir in the ctrl eval layout)
    idx_range : (start, end) 只统计该全局下标区间 (分段并行 eval 用); None = 0..n
    with_lists : True 时额外返回逐样本指标列表 (供跨分段精确合并 std/分位数)
    Returns dict of scalar metrics (with_lists 时返回 (dict, lists_dict)).
    """
    lpips_fn = _get_lpips() if use_lpips else None
    mses, ssims, skels, lpips_ = [], [], [], []
    idxs = list(range(n)) if idx_range is None else list(range(*idx_range))
    for i in idxs:
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
    if with_lists:
        return res, {"idx": idxs, "mse": mses, "ssim": ssims, "skel_iou": skels,
                     "lpips": lpips_}
    return res


# ── eval 条件缓存 (GT 图 + conds + skel + 固定 noise) ───────────────────────
def make_eval_cache(eval_csv, img_root, skel_root, image_size, n,
                    vae_downscale, latent_channels, scaling_factor,
                    skel_latent_shards_dir=None, callig_id_map=None,
                    callig_script_map=None):
    """Pre-load N eval samples: GT images + conditions + skels + fixed noise (CPU).

    skel 条件: 优先从 skel_latent_shards_dir 加载 VAE latent (N,4,32,32);
    否则从 skel_root 读 PNG (N,1,256,256)。两者都存: skels_latent 供采样,
    skels (PNG 时) 供 skel{i}.png 可视化。
    """
    rows = list(csv.DictReader(open(eval_csv, encoding="utf-8")))
    if n > len(rows):
        n = len(rows)
    rows = rows[:n]
    # ★ 2026-09-17: 书家词表校验**必须放在循环之前**。
    #   原来是在循环里 `continue` 掉不在词表里的行 —— 但 `gts`/`skels`/
    #   `skels_latent`/`noise` 都按 `n` 定长，而 `conds` 是循环里 append 的，
    #   跳过一行就少一个 -> `conds` 比 `noise` 短 -> 采样时 batch 不匹配
    #   （实测 x=16 而 y_callig=15，最后在 factorized_cat 的 cat 里报一个
    #    完全看不出原因的 size 错）。
    #   先全量校验：既一次报出所有越界 id，又保证 conds 与数组严格对齐。
    if callig_id_map is not None:
        _miss = sorted({int(r["calligrapher_id"]) for r in rows
                        if int(r["calligrapher_id"]) not in callig_id_map})
        if _miss:
            raise RuntimeError(
                f"[eval-cache] ✗ {eval_csv} 里有 {len(_miss)} 个书家不在 "
                f"callig_id_map 中: {_miss}\n"
                f"  这些行无法映射到模型的书家表（表大小 {len(callig_id_map)}），"
                f"硬跑会退回原始 id -> 索引越界或用错书家（都是静默的）。\n"
                f"  修法: 换一份只含词表内书家的 eval csv，或把词表补全。\n"
                f"  参考 tools/split_eval_from_50k.py（从训练集切 eval，书家必然在表内）。")
    # (书家×书体) 联合风格词表: 设了则 y_callig 用 pair_id。循环前校验每行 pair 可映射
    # (未见 pair 会回退到该书家默认 pair, 不报错但会告警 —— 避免静默用错风格)。
    if callig_script_map is not None:
        _pm = callig_script_map["pair_map"]
        _miss_pair = sorted({f"{int(r['calligrapher_id'])}:{int(r['script_id'])}"
                             for r in rows
                             if f"{int(r['calligrapher_id'])}:{int(r['script_id'])}" not in _pm})
        if _miss_pair:
            print(f"[eval-cache] ⚠ {len(_miss_pair)} 个 (书家,书体) 对不在词表, "
                  f"将回退到该书家默认 pair: {_miss_pair[:10]}")
    transform = T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])
    latent_spatial = image_size // vae_downscale
    gts = torch.zeros(n, 3, image_size, image_size, dtype=torch.float32)
    conds = []
    # S2: 与 conds 平行的三个 id 列表（书家主效应 / pair 残差 / 书体）
    raw_conds, pair_conds, script_conds = [], [], []
    skels = torch.zeros(n, 1, image_size, image_size, dtype=torch.float32)
    skels_latent = torch.zeros(
        n, latent_channels, latent_spatial, latent_spatial, dtype=torch.float32) \
        if skel_latent_shards_dir else None

    # 预建 skel latent shard 索引 (id -> (shard_path, offset))
    skel_id_to_shard = {}
    skel_names = {}
    if skel_latent_shards_dir:
        import glob as _glob
        for sp in sorted(_glob.glob(os.path.join(skel_latent_shards_dir, "shard_*.npz"))):
            with np.load(sp) as d:
                _has_names = "names" in d
                for j, iid in enumerate(d["img_ids"]):
                    skel_id_to_shard[int(iid)] = (sp, j)
                    if _has_names:
                        skel_names[int(iid)] = str(d["names"][j])

    missing_skel = 0
    for i, row in enumerate(rows):
        p = row["image_path"]
        if img_root and not os.path.isabs(p) and not p.startswith(img_root):
            p = os.path.join(img_root, p)
        gts[i] = transform(Image.open(p).convert("RGB"))
        # 书家词表收紧: raw id -> 连续索引 (与训练数据层同一张映射表, 见 callig_map.py)
        _cid = int(row["calligrapher_id"])
        if callig_script_map is not None:
            # (书家×书体) 联合风格词表: y_callig = pair_id (与训练数据层同一张表)
            from src.utils.callig_script_map import map_callig_script
            _cid = map_callig_script(int(row["calligrapher_id"]),
                                     int(row["script_id"]), callig_script_map)
        elif callig_id_map is not None:
            # ⚠ 原来是 `callig_id_map.get(_cid, _cid)` —— **不在词表里就退回原始 id**。
            #   两个后果，都是静默的：
            #     ① 原始 id >= 表大小 -> CUDA 索引越界（v13 在 step5000 就这么崩的：
            #        eval 集里有 5 个书家（39/346/401/483/806）不在 50k 的 45 人词表里，
            #        退回原始 id 806 -> y_callig_embedder(806) 而表只有 46 行）
            #     ② 原始 id < 表大小 -> **静默用错书家**，不报错、指标照出
            #   越界行已在**循环之前**全量校验并报错（见本函数开头），这里直接查表。
            _cid = callig_id_map[_cid]
        conds.append((_cid, int(row.get("glyph_id", row.get("character_id", 0)))))
        # ── S2: 主效应 / pair / 书体 三个 id（与训练数据层**同一套映射**）────
        # ⚠ y_callig 在设了 callig_script_map 时装的是 **pair_id(87)**，
        #   hier 模式下主效应表只有 45 行，必须用 callig_raw，否则越界/串书家。
        _raw_cid = int(row["calligrapher_id"])
        if callig_script_map is not None:
            _cm = callig_script_map.get("callig_map") or {}
            _raw_idx = int(_cm.get(str(_raw_cid), _cm.get(_raw_cid, _raw_cid)))
            _pair_idx = int(_cid)
        elif callig_id_map is not None:
            _raw_idx = int(callig_id_map[_raw_cid])
            _pair_idx = _raw_idx
        else:
            _raw_idx = _raw_cid
            _pair_idx = _raw_cid
        raw_conds.append(_raw_idx)
        pair_conds.append(_pair_idx)
        script_conds.append(int(row.get("script_id", 0)))
        # ★ 2026-09-17: img_id 走统一提取（显式列优先 + 正则锚定结尾 + 失败报错）。
        #   原来 `re.search(r"(\d+)\.png", p)` 未锚定、且失败静默给 None ->
        #   非数字文件名会静默不查 skel（g=ZERO）而不是报错（见 docs/system/70 §1.2）。
        try:
            img_id = extract_img_id(row, where="eval_cache")
        except ValueError as _e:
            if not _ID_WARNED[0]:
                print(f"[eval-cache] ⚠ {_e}")
                _ID_WARNED[0] = True
            img_id = None
        if img_id is not None and skel_names:
            # 与 latent_dataset 同一套内容绑定校验: shard 里的 names 必须等于本行的骨架图名。
            # 没有这道闸，小数据集(如 few-shot)指向大库时会号段撞车 —— 查得到骨架但是
            # 别的字的，评测照样出分，只是分全是错的（2026-09-21 实测作废一整轮实验）。
            _want = os.path.basename(row.get("std_path") or p)
            _got = skel_names.get(int(img_id))
            if _got and _want and _got != _want:
                raise ValueError(
                    f"[eval-cache] skel latent 与 CSV 对不上: img_id={img_id} "
                    f"需要 {_want} 但 shard 里是 {_got} —— "
                    f"检查 eval_skel_latent_shards_dir 是否指向本评测集自己的 shard")
        if img_id is not None and skels_latent is not None:
            if img_id in skel_id_to_shard:
                sp, j = skel_id_to_shard[img_id]
                with np.load(sp, mmap_mode="r") as d:      # ★ mmap: 别解压整个 shard
                    skels_latent[i] = torch.from_numpy(np.array(d["latents"][j], copy=True)).float()
            else:
                missing_skel += 1
        elif img_id is not None and skel_root:
            sk = Image.open(os.path.join(skel_root, f"{img_id}.png")).convert("L")
            sk = sk.resize((image_size, image_size), Image.NEAREST)
            skels[i, 0] = torch.from_numpy(np.asarray(sk, np.float32) / 255.0)
    g = torch.Generator().manual_seed(0)
    noise = torch.randn(n, latent_channels, latent_spatial, latent_spatial, generator=g)
    if skels_latent is not None and missing_skel:
        # ★ 2026-09-17: 从"只打一行 WARNING 继续跑"改为**启动即失败**。
        #   原行为的问题（doc68 §2.2 实际踩过）: shard 目录配错 -> strict 命中 0/237
        #   -> g 全零 -> 字条件完全失效，但评测**照跑不误**，指标看着正常。
        #   早失败 10 秒，胜过跑完 50 步再发现。
        _rate = missing_skel / max(n, 1)
        msg = (f"[eval-cache] ✗ {missing_skel}/{n} ({_rate*100:.1f}%) 样本在 "
               f"{skel_latent_shards_dir!r} 里找不到 skel latent -> 这些样本 g=ZERO"
               f"（字条件失效）")
        if _rate > 0.02:
            raise RuntimeError(
                msg + "\n  覆盖率 <98% 说明 shard 目录配错了（常见：把训练 shard 当成 "
                      "eval shard，或 eval csv 的 id 与 shard 不是同一套）。\n"
                      "  修法: 为 eval csv 单独建 shards（见 config 的 "
                      "eval_skel_latent_shards_dir），或确认 id 一致。\n"
                      "  若确实要容忍少量缺失，请显式降低阈值——不要静默继续。")
        print(msg + "  (比例低, 继续)", flush=True)
    return {"gts": gts, "conds": conds, "noise": noise, "skels": skels,
            "hier_conds": (raw_conds, pair_conds, script_conds),
            "skels_latent": skels_latent, "missing_skel": missing_skel,
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
                             dit_batch, device, skel=skel_arg, seed=0,
                             hier_conds=cache.get("hier_conds"))
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