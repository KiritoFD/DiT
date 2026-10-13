# -*- coding: utf-8 -*-
"""decoder.py — VAE 加载 / 解码（2026-10-13 [审查 §4.1] 从 src/eval/inference.py 迁出）

审查迁移表: "inference.py:355-498 -> decoder.py、metrics.py（**解码器不负责评分**，
            评分器不负责加载 VAE）" ✓

本模块**只**负责"latent -> 像素"这一步所需的全部东西:
  · `match_model_channels` : aux 目标通道时把噪声扩到模型输入通道数（**确定性** ✓）
  · `image_latent`         : 取图像 latent（前 4 通道，丢弃 aux）
  · `load_eval_vae`        : 进程内单例（缓存键 = **(device, 解析后的路径)** ✓）

⚠ 迁移策略（审查"第一阶段只移动逻辑" ✓）: 这些名字仍由 `src/eval/inference.py`
  **原样再导出**（同一个函数对象 ✓）⇒ 全仓 ~25 个既有调用点**一行都不用改** ✓✓。
  缓存单例随函数一起搬到这里 ⇒ 仍然是**唯一**的一份（in_mem_eval 对返回对象打的
  `_VAE.decode` 补丁依旧作用在同一对象上 ✓）。

本模块**不做**评分、不读 GT ✓（那属 metrics/report ✓）。
"""
from __future__ import annotations

import os


__all__ = ["match_model_channels", "image_latent", "load_eval_vae", "VAE_CANDIDATES"]

# ── VAE: 进程内单例 (GPU 推理进程复用) ──────────────────────────────────────
_eval_vae = None
_eval_vae_ref = None

VAE_CANDIDATES = (
    "data/pretrained/pretrained_models/sd-vae-ft-ema",
    "pretrained_models/sd-vae-ft-ema",
    "/root/Workspace/xy/DiT/data/pretrained/pretrained_models/sd-vae-ft-ema",
    "/root/Workspace/xy/DiT/pretrained_models/sd-vae-ft-ema",
)


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
    """Lazily load the VAE once per **(device, 解析后的路径)**。

    ⚠ 2026-10-10 BUGFIX: 原来缓存键**只有 device**，于是同一进程里第二次换 VAE 路径会
    **静默复用第一个 VAE**（不报错、指标也像模像样）。实测后果：先加载 sd-vae-ft-ema，
    再用 Calli-VAE 评测 -> Calli latent 被 SD decoder 解码 -> 整图发灰泛黄
    （mean≈110 vs 正常≈225），LPIPS/SSIM 全部失真。
    这与 `src/eval/in_mem_eval.py` 里 `_VAE_REF # 缓存键 = VAE 路径` 的修复是同一类问题。
    """
    global _eval_vae, _eval_vae_ref
    from diffusers.models import AutoencoderKL
    candidates = [vae_path, *VAE_CANDIDATES]
    resolved = None
    for cand in candidates:
        if cand and os.path.exists(cand):
            resolved = cand
            break
    if not resolved:
        resolved = "stabilityai/sd-vae-ft-ema"
    cache_key = (str(device), str(resolved))          # ★ 路径必须进缓存键
    if _eval_vae is not None and _eval_vae_ref == cache_key:
        return _eval_vae
    _eval_vae = AutoencoderKL.from_pretrained(resolved).to(device).eval()
    for p in _eval_vae.parameters():
        p.requires_grad_(False)
    _eval_vae_ref = cache_key
    return _eval_vae
