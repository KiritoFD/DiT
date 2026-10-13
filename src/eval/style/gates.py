# -*- coding: utf-8 -*-
"""风格编码器的**验收门槛**（方案 §三.2 ✓）—— 与"用什么编码器"无关的纯函数。

三条门槛（可判定化 ✓，与 `tools/style_encoder_accept.py` 的说明一致 ✓）
    G1 2AFC: anchor(w,s,c) 对 正(同书家×同书体×异字) vs 难负(异书家×同书体×同字)
             ⇒ accuracy，**随机 = 0.5**，门槛 **≥0.65** ✓（无需阈值 ✓）
    G2 跨字符书家检索 mAP + **同书体随机基线** ✓
    G3 印刷体负对照: 真实原型下标准字形的书家 top-1 / 平均排名 ✓

为什么抽出来: 训练脚本与验收脚本必须用**同一套**判据 ✓，否则"训练时好看、验收时过不了" ✗。
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List

import numpy as np

from src.eval.style.metrics import cosine, l2norm, cross_char_retrieval


class EmbRows:
    """最小 EmbSet（供 `cross_char_retrieval` 复用 ✓）。"""

    def __init__(self, emb, rows, wmap, smap, cmap, kmap):
        self.emb = np.asarray(emb, dtype=np.float32)
        self._rows = list(rows)
        self._maps = (wmap, smap, cmap, kmap)
        self.writer = np.array([wmap[r["writer"]] for r in rows])
        self.script = np.array([smap[r["script"]] for r in rows])
        self.char = np.array([cmap[r["char"]] for r in rows])
        self.work = np.array([kmap[r["work"]] for r in rows])
        self.name = [str(r.get("path", "")) for r in rows]

    def subset(self, m):
        m = np.asarray(m, dtype=bool)
        idx = np.nonzero(m)[0]
        return EmbRows(self.emb[idx], [self._rows[i] for i in idx], *self._maps)

    def __len__(self):
        return int(self.emb.shape[0])


def _maps(rows):
    f = lambda k: {v: i for i, v in enumerate(sorted({r[k] for r in rows}))}   # noqa: E731
    return f("writer"), f("script"), f("char"), f("work")


def gate1_2afc(emb, rows, *, n_pairs: int = 400, seed: int = 0,
               bootstrap: int = 0) -> Dict[str, float]:
    """同书家异字 vs 异书家同字 —— 2AFC 准确率（0.5 = 随机 ✓）。

    `bootstrap>0` 时额外给 **自助法 95% 置信区间** ✓（对**对子**重采样 ✓）。

    ⚠ 为什么必须给 CI（实测教训 ✓）: n=400 时 1σ ≈ **0.024** ✗ ⇒ 0.6475 vs 门槛 0.65
      完全落在噪声内 ✓ ⇒ 用单点判"过/不过"不严谨 ✓。方案要求"最好在数百组独立对子上
      复核" ✓ ⇒ 本函数把复核变成**可判定**的量 ✓。
    """
    rng = random.Random(seed)
    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[(r["writer"], r["script"], r["char"])].append(i)
    e = l2norm(emb)
    # ★ 2026-10-14 修正（方法学缺陷 ✓）: 原来按 **dict 枚举顺序**取前 n_pairs 个 ✗
    #   ⇒ 估计值依赖枚举顺序与截断点 ✓（实测: 同一编码器 n=1000 得 0.9350、
    #     n=1200 得 0.8833 ✗✗ —— 相差 0.05，纯属**抽样顺序伪影** ✓）。
    #   方案要求"数百组**独立**对子" ✓ ⇒ 必须是**代表性样本** ✓
    #   ⇒ 改为: 先收集**全部可用 anchor 键**并**用 seed 打乱** ✓，再顺序取 n_pairs ✓
    #     （等价于无放回均匀抽样 ✓，且 n_pairs 才真正是样本量 ✓）。
    keys = [k for k in by if any(k2[0] == k[0] and k2[1] == k[1] and k2[2] != k[2] for k2 in by)
            and any(k2[0] != k[0] and k2[1] == k[1] and k2[2] == k[2] for k2 in by)]
    rng.shuffle(keys)
    hits: List[int] = []
    for w, s, c in keys:
        idxs = by[(w, s, c)]
        pos_c = [k for k in by if k[0] == w and k[1] == s and k[2] != c]
        neg_w = [k for k in by if k[0] != w and k[1] == s and k[2] == c]
        a, p = rng.choice(idxs), rng.choice(by[rng.choice(pos_c)])
        n = rng.choice(by[rng.choice(neg_w)])
        hits.append(int(e[a] @ e[p] > e[a] @ e[n]))
        if len(hits) >= n_pairs:
            break
    tried = len(hits)
    out = {"n": tried, "acc": (sum(hits) / tried) if tried else float("nan")}
    if bootstrap and tried:
        br = random.Random(seed + 1)
        arr = np.asarray(hits, dtype=np.float64)
        means = [float(arr[[br.randrange(tried) for _ in range(tried)]].mean())
                 for _ in range(int(bootstrap))]
        out["se"] = float(np.std(means))
        out["ci95"] = [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]
    return out


def gate2_retrieval(emb, rows, *, q_frac: float = 0.3, seed: int = 0) -> Dict[str, float]:
    """跨字符书家检索 mAP（**作品级留出** ✓）+ 同书体随机基线 ✓。"""
    rng = random.Random(seed)
    wmap, smap, cmap, kmap = _maps(rows)
    by_slot = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(rows):
        by_slot[(r["writer"], r["script"])][r["work"]].append(i)
    q = []
    for _slot, works in by_slot.items():
        ws = sorted(works)
        if len(ws) >= 3:
            rng.shuffle(ws)
            for w in ws[:max(1, int(round(len(ws) * q_frac)))]:
                q += works[w]
    q = sorted(set(q))
    if not q:
        return {"map": float("nan"), "n": 0, "base_script": float("nan")}
    E = EmbRows(emb, rows, wmap, smap, cmap, kmap)
    m = np.zeros(len(rows), bool)
    m[q] = True
    r = cross_char_retrieval(E.subset(m), E.subset(~m), topk=(1,))
    g_w, g_s = E.writer[~m], E.script[~m]
    hit = n = 0
    for i in q:
        cand = g_w[g_s == E.script[i]]
        if cand.size == 0:
            continue
        v, cnt = np.unique(cand, return_counts=True)
        hit += int(v[cnt.argmax()] == E.writer[i])
        n += 1
    return {"map": r.get("map", float("nan")), "n": r.get("n_used", 0),
            "base_script": (hit / n) if n else float("nan")}


def gate3_printed(emb_real, rows_real, emb_std, rows_std
                  ) -> Dict[str, float]:
    """印刷体负对照: 真实原型下标准字形的书家 top-1 / 平均排名 ✓。"""
    if emb_std is None or not rows_std:
        return {"skip": 1.0}
    protos: Dict[tuple, np.ndarray] = {}
    for w, s in sorted({(r["writer"], r["script"]) for r in rows_real}):
        idx = [i for i, r in enumerate(rows_real)
               if r["writer"] == w and r["script"] == s]
        if idx:
            protos[(w, s)] = l2norm(emb_real[idx]).mean(0)
    if not protos:
        return {"skip": 1.0}
    keys = list(protos)
    P = l2norm(np.stack([protos[k] for k in keys]))
    Z = l2norm(emb_std)
    hit = n = 0
    ranks: List[int] = []
    for i, r in enumerate(rows_std):
        k = (r["writer"], r["script"])
        if k not in protos:
            continue
        order = np.argsort(-(P @ Z[i]))
        rk = int(np.nonzero(order == keys.index(k))[0][0]) + 1
        ranks.append(rk)
        hit += int(rk == 1)
        n += 1
    return {"n": n, "top1": (hit / n) if n else float("nan"),
            "mean_rank": float(np.mean(ranks)) if ranks else float("nan"),
            "n_writer": len(keys)}


def run_gates(emb, rows, *, emb_std=None, rows_std=None, n_pairs=400, seed=0,
              bootstrap=0, verbose=True) -> Dict[str, object]:
    """跑全部门槛并给出判定 ✓（训练脚本与验收脚本共用 ✓）。"""
    g1 = gate1_2afc(emb, rows, n_pairs=n_pairs, seed=seed, bootstrap=bootstrap)
    g2 = gate2_retrieval(emb, rows, seed=seed)
    g3 = gate3_printed(emb, rows, emb_std, rows_std)
    res = {"G1": g1, "G2": g2, "G3": g3}
    ok1 = g1["acc"] == g1["acc"] and g1["acc"] >= 0.65
    ok2 = (g2["map"] == g2["map"] and g2["base_script"] == g2["base_script"]
           and g2["map"] > g2["base_script"] + 0.05)
    res["pass"] = bool(ok1 and ok2)
    if verbose:
        _ci = g1.get("ci95")
        print("  G1 2AFC(同字异书家) : %.4f  (n=%d%s)   随机 0.5 / 门槛 ≥0.65  -> %s"
              % (g1["acc"], g1["n"],
                 (" 95%%CI [%.4f, %.4f] se=%.4f" % (_ci[0], _ci[1], g1.get("se", 0.0)))
                 if _ci else "",
                 "✓" if ok1 else "✗"))
        print("  G2 跨字符检索 mAP   : %.4f  (n=%d)   同书体基线 %.4f  -> %s"
              % (g2["map"], g2["n"], g2["base_script"], "✓" if ok2 else "✗"))
        if "skip" in g3:
            print("  G3 印刷体负对照     : 跳过（无 std 图 ✓）")
        else:
            print("  G3 印刷体 top1/rank : %.4f / %.2f  (n=%d, 书家数=%d)"
                  % (g3["top1"], g3["mean_rank"], g3["n"], g3["n_writer"]))
    return res


__all__ = ["EmbRows", "gate1_2afc", "gate2_retrieval", "gate3_printed", "run_gates",
           "cosine"]
