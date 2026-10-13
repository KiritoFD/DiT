# -*- coding: utf-8 -*-
"""tests/test_metric_invariants.py — **指标不变量单元测试**（2026-10-13 [审查 §5]）

审查要求: "你们现在的 SSIM/Ink IoU/空白率历史问题，应该给 metric 增加单元测试：
          纯白图、GT=GT、单像素偏移、灰度下降、不同 Unicode 同形，都要产生符合预期的值。"

历史背景（这些测试保护的正是踩过坑的地方）:
  · ssim 曾**奖励退化输出**（v81 白化: 白底大画布 ssim 0.6855 > 真输出 0.6371 ✗）
  · ink_iou 在 **union==0**（两张全白/全空）时按定义返回 **1.0** ✓ —— 所以单看 ink_iou
    分不出"都写对了"和"都没写" ⇒ **必须配 near_blank_rate** ✓（本文件把这个洞固定成测试 ✓）
  · ink 相关指标用 **pred|gt 并集框**, 所以对 1px 平移比全局 ssim 鲁棒得多 ✓（设计意图 ✓）

运行: python -m pytest tests/test_metric_invariants.py -q
      或  python tests/test_metric_invariants.py     （无 pytest 时自跑 ✓）
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.metrics import mse, ssim, skel_iou                      # noqa: E402
from src.eval.metrics_ink import ink_bbox, ink_iou, ink_ssim          # noqa: E402

H = W = 48
INK_THRESH = 0.5          # 与评测口径一致: gray < 0.5 记为墨 ✓
NEAR_BLANK_RATIO = 0.10   # in_mem_eval 的 near_blank 定义: 墨量比 < 0.10 ⇒ 近白 ✓


# ─────────────────────────── 造图工具（不依赖任何真实数据 ✓）───────────────────────────
def white(h=H, w=W):
    """纯白底（无墨）✓"""
    return np.ones((h, w, 1), dtype=np.float32)


def glyph(h=H, w=W, thick=3):
    """一张"有字"的图: 白底 + 一个十字形墨迹（模拟笔画 ✓）"""
    a = white(h, w)
    c = h // 2
    a[c - thick:c + thick, w // 4: 3 * w // 4] = 0.0
    a[h // 4: 3 * h // 4, c - thick:c + thick] = 0.0
    return a


def lighten(a, factor):
    """灰度下降: 墨变浅（0 -> factor, 白保持 1）✓"""
    return 1.0 - (1.0 - a) * factor


def ink_mass(a):
    """墨量比 = mean(1 - gray) —— near_blank 判据用的就是它 ✓"""
    return float(np.mean(1.0 - a))


def near_blank(a):
    return ink_mass(a) < NEAR_BLANK_RATIO


# ─────────────────────────── ① GT = GT（自比, 最强基准 ✓）───────────────────────────
def test_self_identity():
    g = glyph()
    assert abs(mse(g, g)) < 1e-12, "GT 与自身 MSE 必须为 0 ✓"
    assert abs(ssim(g, g) - 1.0) < 1e-6, "GT 与自身 SSIM 必须为 1 ✓"
    assert abs(ink_iou(g, g) - 1.0) < 1e-12, "GT 与自身 ink IoU 必须为 1 ✓"
    assert abs(skel_iou(g, g, thresh=INK_THRESH) - 1.0) < 1e-6, "GT 与自身 skel IoU 必须为 1 ✓"
    assert abs(ink_ssim(g, g) - 1.0) < 1e-6, "GT 与自身 ink-ssim 必须为 1 ✓"


# ─────────────────────────── ② 纯白图（必须被**近白率**检出 ✓✓）───────────────────────────
def test_white_is_detected_by_near_blank_not_by_ssim():
    """审查原话: "白图必被近白率检出" ✓

    关键: 单看 ink_iou **检不出**（都无墨 -> union==0 -> 按定义 1.0 ✓）
          ⇒ 这正是必须同时报 near_blank_rate 的原因 ✓✓
    """
    g, w = glyph(), white()
    assert near_blank(w) is True, "纯白图必须被判为 near_blank ✓"
    assert near_blank(g) is False, "GT 本身不应是 near_blank ✓"
    # 白 vs 有字: 无交集 -> IoU 0 ✓
    assert abs(ink_iou(w, g)) < 1e-12, "纯白 vs GT 的 ink IoU 必须为 0 ✓"


def test_two_blank_images_are_iou_one_the_documented_trap():
    """**两张全白 ⇒ ink_iou = 1.0** ✓✓ —— 文档化的定义, 也是历史上的坑 ✗

    这个测试不是"期望错误", 而是把**已知约定固定下来** ✓: 它告诉我们
    ink_iou 单独用会被"什么都不写"骗过 ⇒ 评测报告必须同时给 near_blank_rate ✓。
    """
    w1, w2 = white(), white()
    assert abs(ink_iou(w1, w2) - 1.0) < 1e-12, (
        "union==0 时 ink_iou 按定义 = 1.0 ✓（本测试把该约定固定; "
        "若今后改成返回 0/NaN **必须同步改评测口径与历史榜单说明** ✗）")
    assert near_blank(w1) and near_blank(w2), "但两者都必须被 near_blank 判出 ✓（互补 ✓）"


# ─────────────────────────── ③ 单像素偏移（墨框设计的意图 ✓）───────────────────────────
def test_shift_sensitivity_and_metric_ordering():
    """平移下的不变量 —— 依据**实测数值表**（不是凭空假设 ✓）

    实测（十字笔画 3px, 48x48）:
        平移   ink_iou  ink_ssim  ssim(gauss)
         0px    1.0000   1.0000    1.0000
         1px    0.8261   0.8225    0.8706
         2px    0.6800   0.6684    0.7913
         4px    0.4483   0.4061    0.7021
         8px    0.2353   0.1862    0.5978

    ⚠ 我先后写错**两个**假设, 都被硬门禁挡住了 ✗✓:
       ① "1px 平移全局 ssim > 0.95" ✗ —— 凭空阈值; 实测 0.87 物理正确 ✓
       ② "墨框口径不低于全局口径" ✗ —— **错**: ink_ssim 用 **box 窗 + 并集框裁剪**,
          与 gauss 全图**本就是两种口径** ⇒ **不可直接比较** ✓（我混淆了 ✓）
    本测试只保留**证据支持**的三条 ✓
    """
    g = glyph()
    # ① 自比满分（口径内部自洽 ✓）
    assert abs(ink_ssim(g, g) - 1.0) < 1e-6 and abs(ink_iou(g, g) - 1.0) < 1e-12
    ks = (0, 1, 2, 4, 8)
    ious = [ink_iou(g, np.roll(g, k, axis=1)) for k in ks]
    inks = [ink_ssim(g, np.roll(g, k, axis=1)) for k in ks]
    sss = [ssim(g, np.roll(g, k, axis=1)) for k in ks]
    # ② 零平移最优 + ink 口径随平移**单调下降**（实测表 ✓）
    for a in range(len(ks) - 1):
        assert ious[a] > ious[a + 1], f"ink_iou 应随平移严格下降: {ious}"
        assert inks[a] > inks[a + 1], f"ink_ssim 应随平移严格下降: {inks}"
    # ③ ★ v81 教训的**量化**: ssim 对墨迹退化**远不如** ink_iou 敏感 ✓✓
    drop_ssim = 1.0 - sss[-1] / sss[0]
    drop_iou = 1.0 - ious[-1] / ious[0]
    assert drop_iou > drop_ssim + 0.2, (
        f"ink_iou 的退化幅度应**显著大于** ssim（这正是必须用墨量类指标的原因 ✓）: "
        f"drop_iou={drop_iou:.3f} drop_ssim={drop_ssim:.3f}")


# ─────────────────────────── ④ 灰度下降（变浅 ⇒ 墨变少 ✓）───────────────────────────
def test_grayscale_reduction():
    g = glyph()
    light = lighten(g, 0.5)                          # 墨由 0 变 0.5 ✓（仍算墨? 0.5 不小于 0.5 ⇒ 不算 ✓）
    assert ink_mass(light) < ink_mass(g), "变浅后墨量比必须下降 ✓"
    assert near_blank(light) or ink_iou(light, g) < 1.0, (
        "变浅要么被判近白、要么 ink IoU 明显低于 1 ✓（不能两者都无变化 ✗）")
    very_light = lighten(g, 0.05)                    # 墨 -> 0.95（几乎白 ✓）
    assert near_blank(very_light) is True, "极度变浅必须被判 near_blank ✓"
    assert abs(ink_iou(very_light, g)) < 1e-9, "几乎全白 vs GT 的 ink IoU 应为 0 ✓"


# ─────────────────────────── ⑤ ink_bbox 的并集语义（防人为放宽 ✓）───────────────────────────
def test_ink_bbox_uses_union_not_gt_only():
    """审查相关: 若只用 GT 的框, pred 跑到框外的部分会被裁掉 = 人为放宽 ✗

    这里造一个"pred 墨迹明显更大"的例子, 断言框**覆盖到 pred** ✓。
    """
    g = glyph()
    big = white()
    big[2:12, 2:12] = 0.0                            # 左上角额外一大块墨（GT 那里是白 ✓）
    y0, y1, x0, x1 = ink_bbox(big, g)
    assert y0 <= 2 and x0 <= 2, f"并集框必须包含 pred 的墨迹（左上角 ✓）, 实得 {(y0, y1, x0, x1)}"
    assert y1 >= 12 and x1 >= 12, "并集框必须同时包含 pred 那块墨 ✓"


def test_ink_bbox_empty_falls_back_to_full_image():
    y0, y1, x0, x1 = ink_bbox(white(), white())
    assert (y0, y1, x0, x1) == (0, H, 0, W), "无墨时返回整图（避免空框 ✗）✓"


# ─────────────────────────── ⑥ 方向性: 越差必须越低（单调性 ✓）───────────────────────────
def test_monotonicity_under_increasing_noise():
    g = glyph()
    prev = 1.0
    for sigma in (0.02, 0.05, 0.10, 0.20):
        noisy = np.clip(g + np.random.RandomState(0).normal(0, sigma, g.shape).astype(np.float32), 0, 1)
        v = ink_ssim(noisy, g)
        assert v <= prev + 1e-6, f"噪声增大时 ink_ssim 不应上升（σ={sigma}: {prev:.4f} -> {v:.4f}）"
        prev = v


# ─────────────────────────── ⑦ SSIM 的口径差异被显式固定 ✓ ───────────────────────────
def test_ssim_window_modes_differ_and_are_both_available():
    """历史上有**两套口径**并存且数值不同 ✓（主路径 gauss / 海报 box）—— 收进一份实现后
    必须保证**两套都还能取到**（否则历史数字对不上 ✗）"""
    g, s = glyph(), np.roll(glyph(), 2, axis=0)
    vg, vb = ssim(g, s, window="gauss"), ssim(g, s, window="box")
    assert 0.0 <= vg <= 1.0 and 0.0 <= vb <= 1.0
    assert abs(vg - vb) > 1e-9, "两套窗口口径的数值本就不同 ✓（若相同说明实现被合并错了 ✗）"
    try:
        ssim(g, s, window="nope")
        raise AssertionError("非法 window 必须报错 ✓")
    except ValueError:
        pass


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    bad = 0
    for fn in fns:
        try:
            fn()
            print("  ✓ %s" % fn.__name__)
        except Exception as e:                                        # noqa: BLE001
            bad += 1
            print("  ✗ %s -> %s: %s" % (fn.__name__, type(e).__name__, str(e)[:110]))
    print("\n  %d/%d 通过" % (len(fns) - bad, len(fns)))
    sys.exit(1 if bad else 0)
