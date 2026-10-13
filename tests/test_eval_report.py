# -*- coding: utf-8 -*-
"""tests/test_eval_report.py — 离线评分层的不变量（2026-10-13 [审查 §4.2]）

审查原话: "生成器只接收 GenerationSample，其中**根本没有 gt_path**。
          只有生成 PNG 落盘之后，评分阶段才获得 GT。"
          "算新指标、重新出 poster，不应该重新调用生成模型。"

本文件把这个"结构性保证"钉成测试 ✓:
  · 第 1 项是**元测试** —— 一旦有人给 GenerationSample 加回任何 GT 字段, 立刻失败 ✓✓
  · "换 GT 只影响指标, 不影响生成身份" 也必须成立 ✓✓

运行: python -m pytest tests/test_eval_report.py -q
"""
from __future__ import annotations

import dataclasses
import os
import sys
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.report import (GenerationSample, ScoringSample, generation_id,   # noqa: E402
                             ink_mass, metrics_id, near_blank, read_samples_jsonl,
                             score_samples, write_samples_jsonl)


def _png(path, arr):
    Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(path)
    return path


def _glyph(h=48, w=48, thick=3):
    a = np.ones((h, w), np.float32)
    c = h // 2
    a[c - thick:c + thick, w // 4: 3 * w // 4] = 0.0
    a[h // 4: 3 * h // 4, c - thick:c + thick] = 0.0
    return a


def _mk(tmp, name, arr):
    return _png(os.path.join(tmp, name), arr)


def _gen(sid="s0", **kw):
    d = dict(sample_id=sid, writer_id=1, script_id=0, char_id=2016, glyph_key="u+6236", seed=7)
    d.update(kw)
    return GenerationSample(**d)


# ─────────── ① ★ 元测试: 生成记录**结构上**不可能带 GT ✓✓ ───────────
def test_generation_sample_has_no_gt_field():
    """审查要求的是**接口上**堵死, 不是"记得别传" ✗ ⇒ 用元测试钉住 ✓✓"""
    names = {f.name for f in dataclasses.fields(GenerationSample)}
    bad = {n for n in names if "gt" in n.lower() or "target" in n.lower() or "label" in n.lower()}
    assert not bad, ("GenerationSample 绝不允许出现 GT/标签字段 ✗✗（审查: 生成器完全不知道 GT）"
                     f"，发现: {sorted(bad)}")
    assert "generated_path" not in names, "生成记录只记**身份**, 不记产物路径 ✓（路径属评分阶段 ✓）"
    # 落盘的 JSONL 也必须只有身份字段 ✓
    tmp = tempfile.mkdtemp()
    p = write_samples_jsonl(os.path.join(tmp, "s.jsonl"), [_gen("a"), _gen("b")])
    txt = open(p, encoding="utf-8").read()
    assert "gt_path" not in txt and ".png" not in txt, "落盘的生成清单里不许出现 GT/图片路径 ✗"


# ─────────── ② 生成身份: 稳定 + 与顺序无关 ✓ ───────────
def test_generation_id_is_canonical():
    a, b = _gen("a"), _gen("b")
    assert generation_id([a, b]) == generation_id([b, a]), "生成身份必须与顺序无关 ✓"
    assert generation_id([a, b]) != generation_id([_gen("a"), _gen("b", seed=8)]), \
        "seed 变化必须改变生成身份 ✓"
    assert read_samples_jsonl(write_samples_jsonl(
        os.path.join(tempfile.mkdtemp(), "s.jsonl"), [a, b])) == [a, b], "JSONL 往返必须等价 ✓"


# ─────────── ③ ★ 换 GT: 指标变, **生成身份不变** ✓✓ ───────────
def test_gt_swap_changes_metrics_but_not_generation_identity():
    tmp = tempfile.mkdtemp()
    g1 = _glyph()
    g2 = np.roll(g1, 6, axis=1)                       # 换一个 GT ✓
    gen = _mk(tmp, "gen.png", g1)
    gt1, gt2 = _mk(tmp, "gt1.png", g1), _mk(tmp, "gt2.png", g2)
    samples = [_gen("s0")]
    gid_before = generation_id(samples)

    m1 = score_samples([ScoringSample("s0", gen, gt1)])
    m2 = score_samples([ScoringSample("s0", gen, gt2)])
    assert m1["samples"][0]["ink_iou"] > m2["samples"][0]["ink_iou"], "换 GT 后指标必须变 ✓"
    assert generation_id(samples) == gid_before, (
        "★ 换 GT **不许**改变生成身份 ✓✓（否则会误判为需要重新生成 ✗）")
    # metrics_id 必须跟着 GT 变（这样才能知道"这份 metrics 是配哪个 GT 算的" ✓）
    assert metrics_id(samples, gt_digest="A") != metrics_id(samples, gt_digest="B")
    # 但换指标集合时, 生成身份同样不变 ✓
    assert generation_id(samples) == gid_before
    assert metrics_id(samples, metric_set=("mse",)) != metrics_id(samples, metric_set=("mse", "ssim"))


# ─────────── ④ 评分确定性 + 白图必被 near_blank 检出 ✓ ───────────
def test_scoring_is_deterministic_and_flags_blank():
    tmp = tempfile.mkdtemp()
    g = _glyph()
    gt = _mk(tmp, "gt.png", g)
    white = _mk(tmp, "white.png", np.ones_like(g))
    same = _mk(tmp, "same.png", g)
    r1 = score_samples([ScoringSample("s0", same, gt)])
    r2 = score_samples([ScoringSample("s0", same, gt)])
    assert r1["samples"][0] == r2["samples"][0], "同输入必须给同结果（确定性 ✓）"
    assert r1["samples"][0]["ink_iou"] > 0.999, "GT vs GT 的 ink_iou 必须为 1 ✓"
    rw = score_samples([ScoringSample("s0", white, gt)])
    row = rw["samples"][0]
    assert row["near_blank_pred"] is True and rw["near_blank_rate"] == 1.0, \
        "纯白生成图**必须**被 near_blank 检出 ✓✓（审查: 白图必被近白率检出）"
    assert row["ink_iou"] < 1e-9, "白 vs 有字 的 ink_iou 必须为 0 ✓"


# ─────────── ⑤ 缺失即停止（不猜 ✗）───────────
def test_missing_files_are_fatal():
    tmp = tempfile.mkdtemp()
    gt = _mk(tmp, "gt.png", _glyph())
    try:
        score_samples([ScoringSample("s0", os.path.join(tmp, "nope.png"), gt)])
        raise AssertionError("生成图缺失必须报错 ✓（不许静默跳过 ✗）")
    except FileNotFoundError:
        pass
    try:
        ScoringSample("s0", "", gt).validate()
        raise AssertionError("空路径必须报错 ✓")
    except ValueError:
        pass


# ─────────── ⑥ 墨量比与近白判据的一致性（与评测口径同一套 ✓）───────────
def test_ink_mass_and_near_blank_agree_with_metric_convention():
    white = np.ones((8, 8, 1), np.float32)
    black = np.zeros((8, 8, 1), np.float32)
    assert ink_mass(white) == 0.0 and near_blank(white) is True
    assert abs(ink_mass(black) - 1.0) < 1e-9 and near_blank(black) is False


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
