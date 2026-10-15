# -*- coding: utf-8 -*-
"""test_refactor_pr2.py —— 2026-10-14 [重构 PR2] REPA 缓存检查 的验收测试 ✓

PR2 完成标准: "数据覆盖 100%，**无静默 fallback**" ✗

审查 §7.1 原文: "`src/utils/dino_cache.py` 当前优先读取 feats.f32 … 而且现有
  `out = buf.to(device=device)` **并没有指定 non_blocking=True** ⇒ 源码注释所说的
  异步 H2D 并没有在这条调用中真正启用" ✗
  并警告: "**不能只把 non_blocking=True 打开就复用同一块 pinned buffer** ✗，
  必须通过双缓冲或 CUDA Event 确保上一次 H2D 完成后才能覆写 ✗"

本组测试锁定:
  ① 文档与实现一致（**同步** H2D ✓ 不再谎称 async ✗）
  ② **无静默 fallback** ✗（pinned 失败默认报错 ✓；memmap 必须**显式** ✓）
  ③ **返回 dtype 恒为 fp32** ✓（修掉 pinned/memmap 不一致的静默 dtype 差异 ✗）
  ④ 覆盖率契约 `require_coverage` ⇒ 缺行**报错** ✗（REPA 是监督信号 ✗）
"""
from __future__ import annotations

import io
import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.dino_cache import DinoFeatureCache                        # noqa: E402

N, P, D = 12, 6, 16


def _mk_cache(tmp_path, dtype="fp32", name="feats.f32"):
    d = str(tmp_path)
    rng = np.random.RandomState(0)
    feats = (rng.randn(N, P, D) * 2).astype(
        np.float32 if dtype == "fp32" else np.float16)
    io.open(os.path.join(d, name), "wb").write(feats.tobytes())
    np.save(os.path.join(d, "ids.npy"), np.arange(100, 100 + N, dtype=np.int64))
    io.open(os.path.join(d, "meta.json"), "w", encoding="utf-8").write(
        json.dumps({"n": N, "patches": P, "dim": D, "backbone": "fake"}))
    return d, feats


# ─────────────── ③ 返回 dtype 一致性（静默差异 ✗ 已修 ✓）───────────────
@pytest.mark.parametrize("disk", ["fp32", "fp16"])
@pytest.mark.parametrize("preload", ["pinned", "memmap"])
def test_gather_always_returns_fp32(tmp_path, disk, preload):
    """★ PR2 ✓: 不论磁盘 dtype 与 preload 模式，**返回恒为 fp32** ✓
    （旧实现 pinned 路径返回磁盘 dtype ✗、memmap 返回 fp32 ✗ ⇒ 静默不一致 ✗✗）"""
    name = "feats.f32" if disk == "fp32" else "feats.f16"
    d, _ = _mk_cache(tmp_path, disk, name)
    c = DinoFeatureCache(d, preload=preload, verbose=False)
    out, miss = c.gather(torch.tensor([100, 103, 111]))
    assert miss == []
    assert out.dtype == torch.float32, f"必须恒为 fp32 ✗ 实际 {out.dtype}"
    assert c.mode == preload


def test_gather_row_correctness(tmp_path):
    d, feats = _mk_cache(tmp_path)
    c = DinoFeatureCache(d, verbose=False)
    out, miss = c.gather(torch.tensor([104, 100]))
    assert miss == []
    assert torch.allclose(out[0], torch.from_numpy(feats[4].astype(np.float32)), atol=1e-6)
    assert torch.allclose(out[1], torch.from_numpy(feats[0].astype(np.float32)), atol=1e-6)


def test_gather_reports_missing(tmp_path):
    d, _ = _mk_cache(tmp_path)
    c = DinoFeatureCache(d, verbose=False)
    out, miss = c.gather(torch.tensor([100, 999, 111]))
    assert miss == [1], miss
    assert out.shape == (3, P, D)


# ─────────────── 可选内存优化: ram_dtype=fp16（默认关闭 ✓）───────────────
def test_ram_dtype_fp16_halves_memory_and_returns_fp32(tmp_path):
    d, feats = _mk_cache(tmp_path)
    c32 = DinoFeatureCache(d, verbose=False)
    c16 = DinoFeatureCache(d, ram_dtype="fp16", verbose=False)
    assert c32._feats.dtype == torch.float32
    assert c16._feats.dtype == torch.float16, "ram_dtype=fp16 必须真的省内存 ✓"
    assert c16._feats.element_size() * 2 == c32._feats.element_size()
    out16, _ = c16.gather(torch.tensor([100, 105]))
    assert out16.dtype == torch.float32, "升回 fp32 ✓"
    assert torch.allclose(out16[0], torch.from_numpy(feats[0].astype(np.float32)),
                          atol=2e-3), "fp16 常驻的数值误差必须很小 ✓"


def test_stats_reports_all_dtypes(tmp_path):
    d, _ = _mk_cache(tmp_path)
    s = DinoFeatureCache(d, ram_dtype="fp16", verbose=False).stats()
    for token in ("disk=fp32", "ram=fp16", "out=fp32", "mode=pinned"):
        assert token in s, (token, s)


def test_invalid_params_rejected(tmp_path):
    d, _ = _mk_cache(tmp_path)
    with pytest.raises(ValueError, match="ram_dtype"):
        DinoFeatureCache(d, ram_dtype="bf16", verbose=False)
    with pytest.raises(ValueError, match="preload"):
        DinoFeatureCache(d, preload="whatever", verbose=False)


# ─────────────── ② 无静默 fallback ✗ ───────────────
def test_pinned_failure_is_fatal_by_default(tmp_path, monkeypatch):
    """★ PR2 ✓: pinned 预载失败 ⇒ 默认**报错** ✗（旧实现只 print 一句就降级 ✗✗）"""
    d, _ = _mk_cache(tmp_path)
    real = torch.from_numpy

    def _boom(arr):
        raise RuntimeError("simulated pinned OOM")
    monkeypatch.setattr(torch, "from_numpy", _boom)
    with pytest.raises(RuntimeError, match="拒绝静默降级"):
        DinoFeatureCache(d, verbose=False)
    monkeypatch.setattr(torch, "from_numpy", real)


def test_pinned_failure_allowed_when_strict_false(tmp_path, monkeypatch):
    """显式 `strict=False` ⇒ 允许回退 ✓（且 mode 如实报告 memmap ✓）"""
    d, _ = _mk_cache(tmp_path)
    real = torch.from_numpy

    def _boom(arr):
        raise RuntimeError("simulated pinned OOM")
    monkeypatch.setattr(torch, "from_numpy", _boom)
    c = DinoFeatureCache(d, strict=False, verbose=False)
    monkeypatch.setattr(torch, "from_numpy", real)
    assert c.mode == "memmap", "回退后必须如实报告 mode ✗"
    out, _ = c.gather(torch.tensor([100]))
    assert out.dtype == torch.float32


# ─────────────── ④ 覆盖率契约（REPA 是监督信号 ✗）───────────────
def test_require_coverage_raises_on_missing(tmp_path):
    d, _ = _mk_cache(tmp_path)
    c = DinoFeatureCache(d, verbose=False)
    assert c.require_coverage([100, 101, 111]) == 3, "全覆盖必须通过 ✓"
    with pytest.raises(KeyError, match="覆盖率不足"):
        c.require_coverage([100, 999], what="REPA 特征")
    with pytest.raises(KeyError):
        c.require_coverage([999])
    assert c.require_coverage([999], raise_on_missing=False) == 0


def test_coverage_query_is_non_destructive(tmp_path):
    d, _ = _mk_cache(tmp_path)
    c = DinoFeatureCache(d, verbose=False)
    assert c.coverage([100, 999, 111]) == (2, 3)


# ─────────────── ① 文档与实现一致（同步 H2D ✓）───────────────
def test_docstring_matches_implementation():
    src = io.open("src/utils/dino_cache.py", encoding="utf-8").read()
    assert "同步 H2D" in src or "**同步** H2D" in src, "必须如实写明是同步 H2D ✓"
    assert "双缓冲" in src, "必须指出真异步需要双缓冲/CUDA Event ✓（审查 §7.1 ✓）"
    assert "不能只打开 non_blocking" in src or "竞态" in src, \
        "必须写明 buf 复用竞态 ✓（审查明确警告 ✓）"
    # 旧文档里那两句谎称 async/无同步的话（**整句**）不得再作为描述出现 ✗
    # 注意: 新文档会**引用**这些旧断言作为历史说明 ✓ ⇒ 只查完整旧句 ✓
    assert "index_select (pinned) -> async H2D。" not in src, \
        "旧文档的 async H2D 断言必须删除 ✗"
    assert "无 page-fault、无 CPU 端 dtype 转换、无同步等待" not in src, \
        "旧文档的『无同步等待』断言必须删除 ✗"
