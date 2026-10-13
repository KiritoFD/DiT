#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""style_encoder_accept.py —— 风格编码器的**验收工具**（方案 §三.2 的三条门槛 ✓）。

方案原文要求（v85 文档 §三.2）::

    | 验收                  | 建议要求                                        |
    | 真实真迹，同字异书家判别 | ≥0.65，最好在数百组独立对子上复核               |
    | 跨字符书家检索         | 明显高于现有编码器及同书体随机基线               |
    | 印刷字体负对照         | 不能再把标准字形高置信度认成某一特定书家          |
    最重要的是在**作品级留出**数据上验证。

三条门槛的可判定化（本工具实现 ✓）
    G1 **2AFC 判别**: 给定 anchor(w, s, c)，两边各取一个候选:
           A = 同书家×同书体×**异字**（正 ✓）   B = **异书家**×同书体×**同字**（难负 ✓）
       判 cos(anchor,A) > cos(anchor,B) 的比例 ⇒ **与随机 0.5 直接可比** ✓（无需阈值 ✓）
       ★ 这正是现有 `cls_style_v1` 报 0.5 的那个量 ✓, 因此可横向对照 ✓
    G2 **跨字符书家检索 mAP** + **同书体随机基线**（只看书体就能猜对书家的比例 ✓）
    G3 **印刷体负对照**: 标准字形渲染图在"书家原型"上的 top-1 命中率与平均排名 ✓
       ⇒ 必须**远低于**真实真迹的同一指标 ✓（否则说明编码器在吃"印刷/工整度" ✗）

用法
    python tools/style_encoder_accept.py --enc dinov2_vitl14 --max-img 3000
    python tools/style_encoder_accept.py --ckpt <encoder.pt> --arch latent
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.style.bank import resolve, work_id                          # noqa: E402
from src.eval.style.metrics import l2norm, cross_char_retrieval            # noqa: E402


def build_rows(csv_path, roots, max_img=0, drop_aug=True):
    """读 CSV 并**均衡抽样** ✓。

    ⚠ 直接按 CSV 顺序截断只会拿到前 1–2 位书家 ✗（CSV 按书家排序 ✓）——
    实测 max_img=2500 时只剩 **3 位书家** ✗ ⇒ 门槛(G1 需要异书家对)根本无从构造 ✓。
    改成**按 (书家,书体) 槽位轮转**取样 ✓, 保证 10 位书家都在 ✓。
    """
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8-sig")))
    by_slot = defaultdict(list)
    miss = 0
    for r in rows:
        if drop_aug and (r.get("aug") or "").strip():
            continue
        p = resolve(r.get("image_path", ""), roots)
        if not p:
            miss += 1
            continue
        d = {"path": p, "writer": (r.get("calligrapher") or "").strip(),
             "script": (r.get("script") or "").strip(),
             "char": (r.get("character") or "").strip(),
             "work": work_id(r.get("src_image_path", ""), r.get("image_path", "")),
             "std": (r.get("std_path") or "").strip()}
        by_slot[(d["writer"], d["script"])].append(d)
    if not max_img:
        out = [d for k in sorted(by_slot) for d in by_slot[k]]
    else:
        out = []
        i = 0
        while len(out) < max_img:
            added = False
            for k in sorted(by_slot):
                if i < len(by_slot[k]):
                    out.append(by_slot[k][i])
                    added = True
                    if len(out) >= max_img:
                        break
            if not added:
                break
            i += 1
    print("[accept] 可用 %d 张（缺图 %d）| 书家 %d | 字 %d | 作品 %d | 槽位 %d"
          % (len(out), miss, len({r['writer'] for r in out}),
             len({r['char'] for r in out}), len({r['work'] for r in out}), len(by_slot)))
    return out


def gate1_2afc(emb, rows, rng, n_pairs=400):
    """同书家异字 vs 异书家同字 —— 2AFC 准确率 ✓（0.5 = 随机 ✓）。"""
    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[(r["writer"], r["script"], r["char"])].append(i)
    e = l2norm(emb)
    hits, tried = 0, 0
    for (w, s, c), idxs in by.items():
        # 同书家×同书体×异字
        pos_c = [k for k in by if k[0] == w and k[1] == s and k[2] != c]
        if not pos_c:
            continue
        # 异书家×同书体×同字
        neg_w = [k for k in by if k[0] != w and k[1] == s and k[2] == c]
        if not neg_w:
            continue
        a = rng.choice(idxs)
        p = rng.choice(by[rng.choice(pos_c)])
        n = rng.choice(by[rng.choice(neg_w)])
        if e[a] @ e[p] > e[a] @ e[n]:
            hits += 1
        tried += 1
        if tried >= n_pairs:
            break
    return {"n": tried, "acc": (hits / tried) if tried else float("nan")}


class _ES:
    """最小 EmbSet 替身（只喂 cross_char_retrieval 需要的字段 ✓）。

    ⚠ subset 必须能重建 —— 我第一版没保存 rows/maps ⇒ `AttributeError: _rows` ✓。
    """
    def __init__(self, emb, rows, wmap, smap, cmap, kmap):
        self.emb = emb
        self._rows = list(rows)
        self._wm, self._sm, self._cm, self._km = wmap, smap, cmap, kmap
        self.writer = np.array([wmap[r["writer"]] for r in rows])
        self.script = np.array([smap[r["script"]] for r in rows])
        self.char = np.array([cmap[r["char"]] for r in rows])
        self.work = np.array([kmap[r["work"]] for r in rows])
        self.name = [os.path.basename(r["path"]) for r in rows]

    def subset(self, m):
        m = np.asarray(m, dtype=bool)
        idx = np.nonzero(m)[0]
        return _ES(self.emb[idx], [self._rows[i] for i in idx],
                   self._wm, self._sm, self._cm, self._km)

    def __len__(self):
        return int(self.emb.shape[0])


def gate2_retrieval(emb, rows, rng, q_frac=0.3):
    """跨字符书家检索 mAP（作品级留出 ✓）+ 同书体随机基线 ✓。"""
    u = lambda k: {v: i for i, v in enumerate(sorted({r[k] for r in rows}))}   # noqa: E731
    wmap, smap, cmap, kmap = u("writer"), u("script"), u("char"), u("work")
    # 作品级留出: 每个槽位里挑一部分**作品**只做 query ✓
    by_slot = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(rows):
        by_slot[(r["writer"], r["script"])][r["work"]].append(i)
    q_idx = []
    for slot, works in by_slot.items():
        ws = sorted(works)
        if len(ws) >= 3:
            rng.shuffle(ws)
            for w in ws[:max(1, int(round(len(ws) * q_frac)))]:
                q_idx += works[w]
    q_idx = sorted(set(q_idx))
    if not q_idx:
        return {"map": float("nan"), "n": 0, "base_script": float("nan")}
    E = _ES(emb, rows, wmap, smap, cmap, kmap)
    m = np.zeros(len(rows), bool)
    m[q_idx] = True
    r = cross_char_retrieval(E.subset(m), E.subset(~m), topk=(1,))
    # 同书体随机基线: 在 gallery 里，只看书体最常出现的书家 = 最naive 的猜法 ✓
    g_w, g_s = E.writer[~m], E.script[~m]
    base_hit, n = 0, 0
    for i in q_idx:
        cand = g_w[g_s == E.script[i]]
        if cand.size == 0:
            continue
        vals, cnt = np.unique(cand, return_counts=True)
        base_hit += int(vals[cnt.argmax()] == E.writer[i])
        n += 1
    return {"map": r.get("map", float("nan")), "n": r.get("n_used", 0),
            "base_script": (base_hit / n) if n else float("nan")}


def gate3_printed(emb_real, rows, emb_std, rows_std, rng):
    """印刷体负对照: 真实原型下, 标准字形的书家 top-1 命中率 / 平均排名 ✓。"""
    if emb_std is None or not rows_std:
        return {"skip": "无 std 渲染图 ✓"}
    protos = {}
    for w, s in sorted({(r["writer"], r["script"]) for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["writer"] == w and r["script"] == s]
        if idx:
            protos[(w, s)] = l2norm(emb_real[idx]).mean(0)
    if not protos:
        return {"skip": "无原型 ✓"}
    keys = list(protos)
    P = l2norm(np.stack([protos[k] for k in keys]))
    Z = l2norm(emb_std)
    hit, ranks = 0, []
    n = 0
    for i, r in enumerate(rows_std):
        k = (r["writer"], r["script"])
        if k not in protos:
            continue
        sim = P @ Z[i]
        order = np.argsort(-sim)
        rk = int(np.nonzero(order == keys.index(k))[0][0]) + 1
        ranks.append(rk)
        hit += int(rk == 1)
        n += 1
    return {"n": n, "top1": (hit / n) if n else float("nan"),
            "mean_rank": float(np.mean(ranks)) if ranks else float("nan"),
            "n_writer": len(keys)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="exp-std/csv/train.csv")
    ap.add_argument("--root", action="append", default=[])
    ap.add_argument("--max-img", type=int, default=3000)
    ap.add_argument("--enc", default="", help="timm 现成编码器（基线对照 ✓）")
    ap.add_argument("--pool", default="cls")
    ap.add_argument("--ckpt", default="", help="自训风格编码器权重（latent 空间的 ✓）")
    ap.add_argument("--device", default="")
    ap.add_argument("--n-pairs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    roots = args.root or [".", "data", "exp-std"]

    rows = build_rows(args.csv, roots, max_img=args.max_img)
    if not rows:
        print("无可用图像 ✗（路径不可解析 ✓）")
        return 2

    if args.enc:
        from src.eval.style.encoders import TimmEncoder
        enc = TimmEncoder.load(args.enc, pool=args.pool, device=(args.device or None))
        emb = enc.embed_paths([r["path"] for r in rows], batch=24, progress=True)[0]
        name = "%s(%s)" % (args.enc, args.pool)
    elif args.ckpt:
        print("latent 编码器验收需要 VAE 产 latent ✗（本机缺 VAE ⇒ 暂不支持 ✓）")
        return 3
    else:
        print("要么给 --enc（基线 ✓），要么给 --ckpt（自训 ✓）")
        return 2

    print("\n===== 验收: %s =====" % name)
    g1 = gate1_2afc(emb, rows, rng, args.n_pairs)
    print("  G1 同字异书家 2AFC : acc=%.4f (n=%d)  ← 随机=0.5, 门槛 **≥0.65**"
          % (g1["acc"], g1["n"]))
    g2 = gate2_retrieval(emb, rows, rng)
    print("  G2 跨字符检索 mAP  : %.4f (n=%d)   同书体随机基线=%.4f  ⇒ 超出 %+.4f"
          % (g2["map"], g2["n"], g2["base_script"], g2["map"] - g2["base_script"]))
    std_rows = []
    for r in rows:
        p = resolve(r["std"], roots)
        if p:
            std_rows.append({"path": p, "writer": r["writer"], "script": r["script"],
                             "char": r["char"], "work": "std", "std": ""})
    emb_std = None
    if std_rows and args.enc:
        from src.eval.style.encoders import TimmEncoder
        emb_std = TimmEncoder.load(args.enc, pool=args.pool,
                                   device=(args.device or None)).embed_paths(
            [r["path"] for r in std_rows], batch=24)[0]
    g3 = gate3_printed(emb, rows, emb_std, std_rows, rng)
    if "skip" in g3:
        print("  G3 印刷体负对照    : %s" % g3["skip"])
    else:
        print("  G3 印刷体 top-1    : %.4f (n=%d, 书家数=%d) 平均排名=%.2f"
              % (g3["top1"], g3["n"], g3["n_writer"], g3["mean_rank"]))
        print("     ⇒ 门槛: 该 top-1 必须**远低**于真实真迹的同一指标 ✓（否则在吃印刷度 ✗）")

    ok1 = g1["acc"] == g1["acc"] and g1["acc"] >= 0.65
    ok2 = g2["map"] == g2["map"] and g2["base_script"] == g2["base_script"] \
        and g2["map"] > g2["base_script"] + 0.05
    print("\n  判定: G1 %s | G2 %s | G3（需与真迹对照后判 ✓）"
          % ("✓ 过" if ok1 else "✗ 不过", "✓ 过" if ok2 else "✗ 不过"))
    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    sys.exit(main())
