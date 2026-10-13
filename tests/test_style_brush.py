# -*- coding: utf-8 -*-
"""Brush-Statistics Distance（BSD）的不变量测试（合成图，不需要真实数据/GPU ✓）。

覆盖:
  ① 已知宽度/角度的合成笔画 ⇒ 宽度、方向特征**确实测到**（不是常数 ✓）
  ② 同图 ⇒ 各分量 = 0
  ③ 加粗 ⇒ width W1 > 0；换角度 ⇒ dir_width JS > 0；变淡 ⇒ gray W1 > 0
  ④ 空白图 ⇒ `ok=False` 且不抛异常（评测里会遇到全白样本 ✓）
  ⑤ 聚合（同书家×同书体一批字）⇒ 分布合并 ✓
"""
from __future__ import annotations

import numpy as np
import pytest

from src.eval.style.brush import (aggregate, bsd, features_from_gray)


def _bar(h=64, w=64, thick=4, angle=0, dark=0.1, jitter=0.0, seed=0):
    """白底 + 一条**已知宽度**的黑色直笔画（angle 单位: 度 ⇒ 0=水平, 90=垂直 ✓）。"""
    img = np.ones((h, w), np.float32)
    img[:, :] = 1.0
    if angle == 0:
        c = h // 2
        img[c - thick // 2:c + thick // 2 + 1, 4:w - 4] = dark
    else:                                    # 垂直
        c = w // 2
        img[4:h - 4, c - thick // 2:c + thick // 2 + 1] = dark
    if jitter:
        rng = np.random.default_rng(seed)
        img = np.clip(img + rng.normal(0, jitter, img.shape).astype(np.float32), 0, 1)
    return img


def test_features_measure_known_width():
    """宽度口径: `2×EDT`（含中心像素）⇒ 比**像素厚度大 1** ✓ —— 生成/参考同约定, 不影响分布距离 ✓。"""
    for thick in (3, 5, 9):
        px = 2 * (thick // 2) + 1               # 我构造的笔画实际像素厚度 ✓
        f = features_from_gray(_bar(thick=thick))
        assert f.ok, "有笔画 ⇒ 特征应可用 ✓"
        st = f.width_stats()
        assert abs(st["p50"] - (px + 1)) <= 1.5, \
            f"thick={thick}(={px}px) 宽度中位数应≈{px + 1}, 实得 {st['p50']:.2f} ✗"
        assert st["p90"] >= st["p10"]
        assert f.width.size >= 8


def test_features_detect_thickness_change():
    thin, thick = features_from_gray(_bar(thick=3)), features_from_gray(_bar(thick=9))
    a = thin.width_stats()["p50"]
    b = thick.width_stats()["p50"]
    assert b > a + 2, f"加粗后宽度中位数必须显著变大（{a:.2f} -> {b:.2f}）✗"
    assert bsd(thin, thick)["width"] > 1.0, "宽度分布距离必须显著 > 0 ✓"


def test_identical_image_zero_distance():
    f = features_from_gray(_bar(thick=5))
    d = bsd(f, f)
    assert d["bsd_macro"] == pytest.approx(0.0, abs=1e-6), "同图 ⇒ 各分量必须为 0 ✓"
    assert d["dir_width_js"] == pytest.approx(0.0, abs=1e-6)


def test_direction_histogram_reacts_to_orientation():
    h = features_from_gray(_bar(angle=0))
    v = features_from_gray(_bar(angle=90))
    assert bsd(h, v)["dir_width_js"] > 0.05, "横/竖笔画的方向×宽度分布必须不同 ✓"
    # 同方向同粗细 ⇒ JS ≈ 0 ✓
    assert bsd(h, features_from_gray(_bar(angle=0)))["dir_width_js"] == pytest.approx(0.0, abs=1e-6)


def test_ink_darkness_affects_gray_and_flywhite():
    dark = features_from_gray(_bar(dark=0.05))
    light = features_from_gray(_bar(dark=0.45))
    assert bsd(dark, light)["gray"] > 0.2, "墨色变淡必须体现在 gray 分量 ✓"
    assert light.scalars["gray_mean"] > dark.scalars["gray_mean"]
    assert light.scalars["flywhite"] >= dark.scalars["flywhite"]


def test_blank_image_is_safe():
    f = features_from_gray(np.ones((64, 64), np.float32))
    assert not f.ok, "全白图 ⇒ ok=False ✓"
    assert np.isnan(bsd(f, f)["bsd_macro"]), "不可计算 ⇒ 返回 NaN 而不是乱给分数 ✓"
    # 与正常图比也不得抛异常 ✓
    g = features_from_gray(_bar())
    assert bsd(f, g)["bsd_macro"] >= 0 or np.isnan(bsd(f, g)["bsd_macro"])


def test_structure_scalars_and_aggregate():
    a = features_from_gray(_bar(h=64, w=64, thick=4))
    b = features_from_gray(_bar(h=32, w=96, thick=4))
    assert "bbox_aspect" in a.scalars and "whitespace" in a.scalars
    agg = aggregate([a, b])
    assert agg.ok and agg.width.size == a.width.size + b.width.size, "聚合必须合并分布 ✓"
    assert agg.scalars["ink_frac"] == pytest.approx(
        np.mean([a.scalars["ink_frac"], b.scalars["ink_frac"]]))
    assert aggregate([]).ok is False, "空输入不得抛异常 ✓"


def test_rgba_and_uint8_paths_agree(tmp_path=None):
    """uint8 与 float 输入必须得到同样的特征（极性/量纲统一 ✓）。"""
    img = _bar(thick=4)
    f_float = features_from_gray(img)
    f_uint8 = features_from_gray((img * 255).astype(np.uint8))
    assert f_uint8.width_stats()["p50"] == pytest.approx(f_float.width_stats()["p50"], abs=0.6)
    # RGBA: 透明背景 + 黑字 ⇒ 与白底黑字同构 ✓
    rgba = np.zeros((64, 64, 4), np.uint8)
    ink = img < 0.5
    rgba[..., 3] = np.where(ink, 255, 0)
    f_rgba = features_from_gray(rgba)
    assert f_rgba.ok and f_rgba.width_stats()["p50"] == pytest.approx(
        f_float.width_stats()["p50"], abs=1.0), "RGBA 透明背景必须被当作白底 ✓"
