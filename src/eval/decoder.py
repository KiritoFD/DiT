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

import numpy as np
import torch
from PIL import Image

# ⚠ 2026-10-13: 下面三个顶层 import 是 **decode_and_save 迁入后**才必需的 ✓。
#   首版 decoder 只含 load_eval_vae 时, 顶层 `import torch` 被 ruff --fix 判为 F401 删掉 ✗
#   （当时 torch 只在 match_model_channels 内以 `import torch as _t` 局部使用）⇒ 迁入后
#   `@torch.no_grad()` 立刻 NameError ✓✗。教训: **--fix 之后必须重跑受影响检查** ✓。


__all__ = ["match_model_channels", "image_latent", "load_eval_vae", "decode_and_save",
           "get_white_latent", "maybe_add_white", "VAE_CANDIDATES"]

# ── 白底归零用的常量（随 get_white_latent/maybe_add_white 一起迁入 ✓）──
_WHITE_LAT = {}

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


# ── VAE decode (fp32) -> PNG 落盘 ────────────────────────────────────────────
#   2026-10-13 [审查 §4.1]: 从 src/eval/inference.py 迁入（"解码器不负责评分" ✓）。
#   ⚠ 迁移时**必须连装饰器一起搬** —— AST 的 FunctionDef.lineno 指向 def 行, 从那里切会把
#     @torch.no_grad() 留在原处（解码就会在开梯度下跑 ✗）。本文件按 decorator 行切 ✓。
@torch.no_grad()
def decode_and_save(vae, latents, scaling_factor, out_dir, tag, conds=None,
                    gts=None, vae_batch=16, skels=None, idx_offset=0,
                    zero_white=False, img_channels=4):
    # ★ 2026-10-05: img_channels 显式化 —— 16ch Flux VAE 下图像 latent 是 16 通道,
    #   原来硬编码 `lat[:, :4]` 会让 16ch VAE 解码失败。默认 4 保持向后兼容。
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
        if lat.shape[1] > img_channels:      # aux 目标通道: 解码只用图像通道
            lat = lat[:, :img_channels]
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


# ── 白底归零 (aux_zero_white) 的**唯一**加回入口 ──────────────────────────────
#   2026-10-13 [审查 §4.1]: 与 decode_and_save 一起迁入 —— 后者调用 maybe_add_white,
#   若把它留在 inference.py 就会让 decoder **反向依赖** inference 而成环 ✗。
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
