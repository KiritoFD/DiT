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

import torch


_ID_WARNED = [False]


# ── VAE 加载/解码已迁出（2026-10-13 [审查 §4.1]: 解码器不负责评分, 评分器不负责加载 VAE）──
#   ★ 迁移策略: 下面三个名字在此**原样再导出**（同一个函数对象 ✓）⇒ 全仓 ~25 个既有调用点
#     **一行都不用改** ✓✓；VAE 缓存单例随函数一起搬走, 仍是唯一一份 ✓（in_mem_eval 对返回
#     对象打的 _VAE.decode 补丁依旧作用在同一对象上 ✓）。新代码请直接 from src.eval.decoder import …
from src.eval.decoder import (decode_and_save, get_white_latent, image_latent,   # noqa: F401
                              load_eval_vae, match_model_channels, maybe_add_white)


# ── 评测缓存已迁出（2026-10-13 [审查 §4.1]: 合并评测样本映射和缓存）────────────────────
from src.eval.dataset import make_eval_cache   # noqa: F401  （原样再导出, 既有调用点无需改动 ✓）

# ── 采样实现已迁出（2026-10-13 [审查 §4.1]: 保留唯一 Heun/SDE/CFG 实现）──────────────────
#   ★ 同样**原样再导出**（同一函数对象 ✓）⇒ 既有调用点无需改动 ✓；新代码请 from src.eval.sampler import …
from src.eval.sampler import (build_diffusion, model_uses_std_soft,   # noqa: F401
                              sample_latents, sample_latents_self_cond,
                              validate_std_soft_condition)

# ── 目录级评分器已迁出（2026-10-13 [审查 §4.1]: 评分器不负责加载 VAE）──────────────────
#   含 _mse/_ssim/_skel_iou 三个私有别名（有工具直接 import ✓）与 _get_lpips ✓
from src.eval.scoring import (_get_lpips, _mse, _skel_iou, _ssim,   # noqa: F401
                              compute_metrics)


# ── 白底归零 (aux_zero_white) 的**唯一**加回入口 ──────────────────────────────




# ── VAE decode (fp32) → PNG 落盘 ────────────────────────────────────────────


# ── CPU 指标 ────────────────────────────────────────────────────────────────
# ★ 2026-09-17: 实现已统一到 src/eval/metrics.py（仓库里曾有 14 份 SSIM / 8 份
#   skel_iou / 4 份 MSE 的复制粘贴，且 win 与输入约定不一致 -> 数字不可比）。
#   这里保留 `_mse` / `_ssim` / `_skel_iou` 三个名字做**向后兼容的再导出**，
#   所有既有调用点无需改动。口径与历史主评测路径完全一致（win=11 高斯窗）。




# ── eval 条件缓存 (GT 图 + conds + skel + 固定 noise) ───────────────────────


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
