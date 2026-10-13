# -*- coding: utf-8 -*-
"""nearest_neighbor.py — **同字最近邻检索**（2026-10-13 [审查 §4.1] 从 in_mem_eval.py 迁出）

审查: "同字最近邻检索改为**可选离线分析, 不占每次训练内评测**" ✓

⚠ **有意偏离审查的一处**: 不放 tools/ ✗ —— in_mem_eval 在开关打开时仍要调它 ✓, 放 tools/ 会
  造成 src/ 反向依赖 	ools/ ✗（违反本仓 src/ 纯净约定 ✓）⇒ 落点 src/eval/nearest_neighbor.py ✓。
  审查要的"不占每次训练内评测"由**调用处默认关闭**实现 ✓。

本模块不依赖 in_mem_eval ✓（单向 ✓）；离线工具可直接 import ✓。
"""
from __future__ import annotations

import csv
import os

import numpy as np
from PIL import Image

# _ssim 与 in_mem_eval 里**同一实现**（原文件即从 metrics 取 ✓）—— 搬移时按原文口径补上 ✓。
from src.eval.metrics import ssim as _ssim

__all__ = ["samechar_nn_eval", "_SAMECHAR_BYCHAR"]


_SAMECHAR_BYCHAR = {}


def samechar_nn_eval(pred_np, gt_np, ev_rows, train_csv, max_cand=12, nw=16):
    """同字最近邻指标 —— 直接回答"生成结果最像训练集里谁写的"。

    对每一列, 在该字的训练集候选里找 SSIM 最高的那张 GT:
      nn_ssim  : max SSIM (均值 over 列)            "最像谁"
      nn_mean  : 同字候选的 SSIM 均值                "随机挑一张的期望"
      tgt_spec : own_gt - nn_mean                   **目标特异性**
      cal_hit/cal_base/cal_enrich                   有同书家候选时命中同书家 / 基线 / 富集倍数

    ⚠ 两个必须一起看才不误读的点:
      1. nn_ssim - nn_mean 天然 >0 (max≥mean), 不能单独当判别力证据 -> 要看 tgt_spec
      2. "最近邻是同书家"的比例会被"有多少列**有**同书家候选"限制 -> 必须看条件命中率与基线
    """
    import concurrent.futures as _cf
    if not (train_csv and len(pred_np) and ev_rows):
        return None
    if train_csv not in _SAMECHAR_BYCHAR:
        import collections as _co
        _by = _co.defaultdict(list)
        try:
            with open(train_csv, encoding="utf-8") as _f:
                for _r in csv.DictReader(_f):
                    _by[str(_r.get("character", ""))].append(_r)
        except Exception as _e:
            print(f"[samechar-nn] 训练 csv 不可读 ({_e!r})，跳过")
            _SAMECHAR_BYCHAR[train_csv] = None
            return None
        _SAMECHAR_BYCHAR[train_csv] = _by
    by_char = _SAMECHAR_BYCHAR[train_csv]
    if by_char is None:
        return None

    def _one(i):
        if i >= len(ev_rows):
            return None
        r = ev_rows[i]
        ch = str(r.get("character", ""))
        cal = r.get("calligrapher")
        cs = by_char.get(ch, [])
        if not cs:
            return None
        if len(cs) > max_cand:
            _st = len(cs) / max_cand
            cs = [cs[int(k * _st)] for k in range(max_cand)]
        own = float(_ssim(pred_np[i], gt_np[i]))
        ss = []
        for c in cs:
            p = c.get("image_path", "")
            if not p:
                continue
            if not os.path.isabs(p):
                p = os.path.join(os.getcwd(), p)
            if not os.path.exists(p):
                continue
            # ⚠ 模块级是 `from PIL import Image`（没有 _Img 别名，_Img 只在 render_poster 内定义）。
            #   之前这里写 _Img.open 抛 NameError, 又被 `except Exception: continue` **静默吞掉**
            #   -> 所有候选被跳过 -> 函数返回 None, 不报错。这就是本项目反复出现的失败模式。
            #   现在只吞 IO 错误, 其它异常照常抛出。
            try:
                with Image.open(p) as _f:
                    t = np.asarray(_f.convert("RGB"), dtype=np.float32) / 255.0
            except (OSError, ValueError):
                continue
            if t.shape != pred_np[i].shape:
                continue
            ss.append((float(_ssim(pred_np[i], t)), c.get("calligrapher")))
        if not ss:
            return None
        vals = [s for s, _ in ss]
        ncal = sum(1 for _, c2 in ss if c2 == cal)
        hit = bool(ncal) and (max([s for s, c2 in ss if c2 == cal]) >= max(vals) - 1e-12)
        return dict(own=own, nn=max(vals), mc=float(np.mean(vals)),
                    ncal=ncal, n=len(ss), hit=hit)

    with _cf.ThreadPoolExecutor(max_workers=max(1, min(nw, len(pred_np)))) as _ex:
        rs = [x for x in _ex.map(_one, range(len(pred_np))) if x]
    if not rs:
        return None
    av = [x for x in rs if x["ncal"] > 0]
    _hit = (sum(1 for x in av if x["hit"]) / len(av)) if av else float("nan")
    _base = (sum(x["ncal"] / max(x["n"], 1) for x in av) / len(av)) if av else float("nan")
    return dict(n_ok=len(rs),
                nn_ssim=float(np.mean([x["nn"] for x in rs])),
                nn_mean=float(np.mean([x["mc"] for x in rs])),
                tgt_spec=float(np.mean([x["own"] - x["mc"] for x in rs])),
                cal_hit=_hit, cal_base=_base,
                cal_enrich=(_hit / _base) if (av and _base and _base == _base and _base > 0)
                else float("nan"),
                n_cal=len(av))

_samechar_nn_eval = samechar_nn_eval   # 旧名别名 ✓
