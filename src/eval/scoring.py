# -*- coding: utf-8 -*-
"""scoring.py — **目录级评分器**（2026-10-13 [审查 §4.1] 从 src/eval/inference.py 迁出）

审查迁移表: "inference.py:355-498 -> decoder.py、metrics.py（解码器不负责评分,
            **评分器不负责加载 VAE**）" ✓

⚠ **有意偏离审查的一处**（已在提交里说明）: 审查写的是并入 metrics.py ✓，但本模块含
  _get_lpips() —— 一个**加载学习型指标模型**(lpips vgg)的入口 ✗。按审查自己那条原则
  （"评分器不负责加载 VAE" ⇒ 同类地, 纯指标实现不该夹带模型加载 ✗）, 这里**另立 scoring.py**:
      · src/eval/metrics.py  : **纯**数值指标（无任何模型加载 ✓ 已有 9 项不变量测试 ✓）
      · src/eval/scoring.py  : 目录级评分（读 PNG -> 指标 dict ✓）+ **可选** LPIPS ✓
  这样"纯指标"与"会加载模型的评分"职责分离 ✓；若审查更希望合并, 只需把本文件内容移入 metrics.py ✓。

⚠ 迁移策略（审查"第一阶段只移动逻辑" ✓）: 五个名字（含 _mse/_ssim/_skel_iou 私有别名,
  有工具直接 import ✓）仍由 src/eval/inference.py **原样再导出** ✓。
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image

from src.eval.metrics import mse as _mse_impl, ssim as _ssim_impl, \
    skel_iou as _skel_iou_impl

__all__ = ["compute_metrics", "_mse", "_ssim", "_skel_iou", "_get_lpips"]


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
