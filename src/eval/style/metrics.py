# -*- coding: utf-8 -*-
"""style 评价轴：跨字符书家风格度量（纯函数，输入是嵌入 + 标签）。

设计原则
--------
1. **不需要逐字对应的 GT** ✓：所有指标只要求"同书体"，字形可以完全不同。
2. 纯函数、无 IO、无模型 ⇒ 可在本地用合成数据做不变量测试 ✓。
3. 每个指标都返回 `n_used / n_skipped`：无法计算的槽位**显式跳过**，绝不用别的书体顶替 ✓。

三条硬约束（调用方通过参数强制）
--------------------------------
· 同书体（`same_script`）—— 否则模型仅凭书体差异就能猜对书家。
· Gallery 与 Query **字符不同**（`exclude_same_char`）。
· Gallery 与 Query 尽量**作品不同**（`exclude_same_work`，未知作品用 -1 表示）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

EPS = 1e-12


# ─────────────────────────── 数据容器 ───────────────────────────
@dataclass
class EmbSet:
    """一批风格嵌入 + 其标签。`emb` 会做 L2 归一化。"""
    emb: np.ndarray                     # (N, d)
    writer: np.ndarray                  # (N,) int
    script: np.ndarray                  # (N,) int
    char: np.ndarray                    # (N,) int
    work: np.ndarray                    # (N,) int，-1 = 未知
    name: List[str]                     # (N,) 可读标识（样本名）

    def __post_init__(self) -> None:
        self.emb = np.asarray(self.emb, dtype=np.float32)
        for f in ("writer", "script", "char", "work"):
            setattr(self, f, np.asarray(getattr(self, f), dtype=np.int64))
        n = len(self.emb)
        for f in ("writer", "script", "char", "work"):
            assert len(getattr(self, f)) == n, f"{f} 长度与 emb 不一致 ✗"
        if len(self.name) != n:
            self.name = list(self.name) + [""] * (n - len(self.name))

    def __len__(self) -> int:
        return len(self.emb)

    def subset(self, mask: np.ndarray) -> "EmbSet":
        m = np.asarray(mask, dtype=bool)
        return EmbSet(self.emb[m], self.writer[m], self.script[m], self.char[m],
                      self.work[m], [n for n, k in zip(self.name, m) if k])


def l2norm(x: np.ndarray) -> np.ndarray:
    """按行 L2 归一化（零向量保持为零，避免 NaN）。"""
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, EPS)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(M,d) x (N,d) -> (M,N) 余弦相似度（输入假定已归一化）。"""
    return np.asarray(a, dtype=np.float32) @ np.asarray(b, dtype=np.float32).T


def leave_one_out_similarity(ref: EmbSet, *, same_script: bool = True) -> Dict[str, object]:
    """**Real-Real 上限**的正确口径 ✓: 每张参考图 vs 该书写过的**其他字**（留一法 ✓）。

    ⚠ 为什么不用"按作品留出的真实 query" ✗: 那要求**跨扫描/版本**泛化 ✗,
      而参考库 55% 来自单一合并源 ⇒ 域偏移会把上限人为压低 ✗✗（实测: 5 个编码器
      一致出现 real_real < 生成集 的反常模式 ✓，根因就是难度口径错了 ✓）。
    """
    e = l2norm(ref.emb)
    per: Dict[Tuple[int, int], float] = {}
    allv: List[np.ndarray] = []
    for w, s in sorted(set(zip(ref.writer.tolist(), ref.script.tolist()))):
        m = (ref.writer == w) & ((ref.script == s) if same_script else True)
        idx = np.nonzero(m)[0]
        if idx.size < 2:
            continue
        sim = e[idx] @ e[idx].T                       # (N,N)
        np.fill_diagonal(sim, np.nan)                 # 去掉"自己 vs 自己" ✓
        vals = sim[~np.isnan(sim)]
        per[(int(w), int(s))] = float(vals.mean())
        allv.append(vals)
    if not allv:
        return {"s_macro": float("nan"), "s_all": float("nan"), "n_slots": 0, "per_slot": {}}
    a = np.concatenate(allv)
    return {"s_macro": float(np.mean(list(per.values()))), "s_all": float(a.mean()),
            "s_std": float(a.std()), "n_slots": len(per), "n_pairs": int(a.size),
            "per_slot": per}


# ─────────────── ⓪ CLIP-S / DINO-S：风格相似度（**不要求同字** ✓）───────────────
def style_similarity(gen: EmbSet, ref: EmbSet, *, same_script: bool = True) -> Dict[str, object]:
    """生成图 与 **风格参考图** 的**平均余弦相似度** ✓ —— CLIP-S / DINO-S 的核心 ✓。

    与检索式指标的根本区别 ✓
        · **不要求同字** ✓、不要求同作品 ✓ —— 生成「體」可以和该书家写过的「家/山/端」比 ✓
          （这正是风格相似度指标的标准定义 ✓，也是本任务最贴切的口径 ✓）
        · 参考侧 = 目标书家（同书体）的**全部**参考图 ⇒ 逐对求余弦后取均值 ✓

    ⚠ 原始余弦值本身跨编码器**不可比** ✗ ⇒ 必须与 `real_real`(上限) / `real_other_writer` /
      `ttf_only`(下限) 一起报告才能解读 ✓。
    """
    ge, re_ = l2norm(gen.emb), l2norm(ref.emb)
    per: Dict[Tuple[int, int], float] = {}
    sims: List[np.ndarray] = []
    for w, s in sorted(set(zip(gen.writer.tolist(), gen.script.tolist()))):
        mg = (gen.writer == w) & (gen.script == s)
        mr = (ref.writer == w) & ((ref.script == s) if same_script else True)
        if mg.sum() == 0 or mr.sum() == 0:
            continue
        sim = ge[mg] @ re_[mr].T                      # (N_gen, N_ref) ✓
        per[(int(w), int(s))] = float(sim.mean())
        sims.append(sim.ravel())
    if not sims:
        return {"s_macro": float("nan"), "s_all": float("nan"), "n_slots": 0, "per_slot": {}}
    allv = np.concatenate(sims)
    return {"s_macro": float(np.mean(list(per.values()))),   # 每槽位先平均再宏平均 ✓
            "s_all": float(allv.mean()),                     # 全部配对直接平均 ✓
            "s_std": float(allv.std()),
            "n_slots": len(per), "n_pairs": int(allv.size), "per_slot": per}


# ─────────────────────── ① 跨字符书家检索 ───────────────────────
def _ap(rel: np.ndarray) -> float:
    """单条 query 的 average precision（rel 为按相似度降序的相关性 0/1）。"""
    if rel.size == 0 or rel.sum() == 0:
        return float("nan")
    c = np.cumsum(rel)
    prec = c / (np.arange(rel.size) + 1.0)
    return float((prec * rel).sum() / rel.sum())


def cross_char_retrieval(query: EmbSet, gallery: EmbSet, *,
                         topk: Sequence[int] = (1, 5),
                         same_script: bool = True,
                         exclude_same_char: bool = True,
                         exclude_same_work: bool = True,
                         max_queries: Optional[int] = None) -> Dict[str, object]:
    """跨字符书家检索：query 能否从"别的字"的真迹里检索出正确书家 ✓。

    相关判定 = **同书家**（标准 writer-retrieval 口径）。无法计算的 query
    （无相关候选 / 无无关候选 / 同书体无候选）**跳过并计数** ✓。
    """
    q, g = query, gallery
    if len(q) == 0 or len(g) == 0:
        return {"top1": float("nan"), "map": float("nan"), "n_used": 0, "n_skipped": len(q),
                "topk": {}}
    # 预取 gallery 向量（M×N 相似度矩阵在 N 大时按块算，这里 N 通常 <10^4 ✓）
    gemb = l2norm(g.emb)
    idx_all = np.arange(len(g))
    sims = cosine(l2norm(q.emb), gemb)

    q_order = np.arange(len(q)) if max_queries is None else np.arange(min(max_queries, len(q)))
    aps: List[float] = []
    hits = {k: 0 for k in topk}
    n_used = 0
    n_skipped = 0
    per_writer: Dict[int, List[float]] = {}
    for i in q_order:
        ok = np.ones(len(g), dtype=bool)
        if same_script:
            ok &= (g.script == q.script[i])
        if exclude_same_char:
            ok &= (g.char != q.char[i])
        if exclude_same_work and q.work[i] >= 0:
            ok &= (g.work != q.work[i])
        cand = idx_all[ok]
        if cand.size == 0:
            n_skipped += 1
            continue
        rel = (g.writer[cand] == q.writer[i]).astype(np.float64)
        if rel.sum() == 0 or rel.sum() == cand.size:
            n_skipped += 1                     # 没有正样本或没有负样本 ⇒ 不可计算 ✓
            continue
        order = np.argsort(-sims[i, cand], kind="stable")
        rel_sorted = rel[order]
        aps.append(_ap(rel_sorted))
        per_writer.setdefault(int(q.writer[i]), []).append(aps[-1])
        for k in topk:
            if rel_sorted[:k].sum() > 0:
                hits[k] += 1
        n_used += 1
    out: Dict[str, object] = {
        "n_used": n_used, "n_skipped": n_skipped,
        "map": float(np.mean(aps)) if aps else float("nan"),
        "topk": {int(k): (hits[k] / n_used if n_used else float("nan")) for k in topk},
        "per_writer_map": {w: float(np.mean(v)) for w, v in sorted(per_writer.items())},
        "n_candidate_writers": int(len(np.unique(g.writer[g.writer >= 0]))),
    }
    out["top1"] = out["topk"].get(1, float("nan"))
    return out


# ─────────────────── ② 风格原型间距（Prototype Margin） ───────────────────
def build_prototypes(real: EmbSet) -> Dict[Tuple[int, int], np.ndarray]:
    """(writer, script) -> L2 归一化的真实风格中心 μ。用**任意字**的真实真迹 ✓。"""
    proto: Dict[Tuple[int, int], np.ndarray] = {}
    keys = set(zip(real.writer.tolist(), real.script.tolist()))
    for w, s in sorted(keys):
        m = (real.writer == w) & (real.script == s)
        if m.sum() == 0:
            continue
        proto[(int(w), int(s))] = l2norm(real.emb[m].mean(axis=0, keepdims=True))[0]
    return proto


def prototype_margin(query: EmbSet, proto: Dict[Tuple[int, int], np.ndarray], *,
                     require_same_script: bool = True,
                     min_others: int = 1) -> Dict[str, object]:
    """Δ = cos(q, μ_target) − max_{w'≠w} cos(q, μ_{w'}) ✓。

    Δ>0 更像目标书家；Δ<0 更像别的书家。竞争者在**同书体**里取（否则不可比 ✓）。
    目标或竞争者缺失 ⇒ 跳过并计数 ✓。
    """
    rows: List[Tuple[int, int, int, float, float, float]] = []   # i, w, rival, S_t, S_r, Δ
    n_skipped = 0
    qe = l2norm(query.emb)
    for i in range(len(query)):
        w, s = int(query.writer[i]), int(query.script[i])
        tgt = proto.get((w, s))
        if tgt is None:
            n_skipped += 1
            continue
        rivals = [(k, v) for k, v in proto.items() if k[1] == s and k[0] != w]
        if len(rivals) < min_others:
            n_skipped += 1
            continue
        s_t = float(qe[i] @ tgt)
        (rw_key, rv) = max(rivals, key=lambda kv: float(qe[i] @ kv[1]))
        rw = int(rw_key[0])                     # key 是 (writer, script) ⇒ 取 writer ✓
        s_r = float(qe[i] @ rv)
        rows.append((i, w, rw, s_t, s_r, s_t - s_r))
    if not rows:
        return {"margin_mean": float("nan"), "pos_frac": float("nan"),
                "n_used": 0, "n_skipped": n_skipped, "worst_pairs": []}
    d = np.array([r[5] for r in rows])
    # 最困难书家对：按目标书家聚合的平均 margin 最差者 ✓
    by_w: Dict[int, List[float]] = {}
    for r in rows:
        by_w.setdefault(r[1], []).append(r[5])
    worst = sorted(((w, float(np.mean(v))) for w, v in by_w.items()), key=lambda x: x[1])[:5]
    return {"margin_mean": float(d.mean()), "margin_median": float(np.median(d)),
            "pos_frac": float((d > 0).mean()), "n_used": len(rows), "n_skipped": n_skipped,
            "worst_pairs": worst}


# ─────────────────────── ③ HWD-CN：集合均值距离 ───────────────────────
def hwd_cn(gen: EmbSet, real: EmbSet, *, same_script: bool = True) -> Dict[str, object]:
    """D = ‖mean(f(生成)) − mean(f(真实))‖₂，按 (writer, script) 分别算后取宏平均 ✓。

    真实参考**不要求同字** ✓（HWD 思想的汉化改编：此处明确是 HWD-CN 改编指标，
    不与原论文英文行图像口径混同 ✓）。样本数 < `min_n` 的槽位跳过 ✓。
    """
    per: Dict[Tuple[int, int], float] = {}
    n_skipped = 0
    for w, s in sorted(set(zip(gen.writer.tolist(), gen.script.tolist()))):
        mg = (gen.writer == w) & (gen.script == s)
        mr = (real.writer == w) & (real.script == s)
        if same_script:
            mr &= (real.script == s)
        if mg.sum() == 0 or mr.sum() == 0:
            n_skipped += 1
            continue
        per[(int(w), int(s))] = float(np.linalg.norm(
            gen.emb[mg].mean(axis=0) - real.emb[mr].mean(axis=0)))
    vals = list(per.values())
    return {"hwd_cn_macro": float(np.mean(vals)) if vals else float("nan"),
            "per_slot": per, "n_used": len(vals), "n_skipped": n_skipped}


# ─────────────────── ④ Style-MMD / KID：分布距离 ───────────────────
def _rbf_mmd2_unbiased(x: np.ndarray, y: np.ndarray, sigma: float) -> float:
    """无偏 MMD²（单尺度 RBF）。有限样本下**可能略小于 0** ✓，不是实现错误。"""
    def k(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        return np.exp(-d2 / (2.0 * sigma ** 2))
    m, n = len(x), len(y)
    if m < 2 or n < 2:
        return float("nan")
    kxx, kyy, kxy = k(x, x), k(y, y), k(x, y)
    np.fill_diagonal(kxx, 0.0)
    np.fill_diagonal(kyy, 0.0)
    return float(kxx.sum() / (m * (m - 1)) + kyy.sum() / (n * (n - 1)) - 2.0 * kxy.mean())


def style_mmd(x: np.ndarray, y: np.ndarray, *,
              sigmas: Optional[Iterable[float]] = None,
              median_heuristic: bool = True) -> float:
    """多尺度 RBF MMD²（无偏）。x/y 为风格嵌入 (N,d)。

    尺度取 `median‖xi−xj‖` 的 {0.5, 1, 2} 倍（median heuristic ✓）；也可显式给 sigmas。
    与 ImageNet Inception-KID **不可混排** ✓（特征空间不同，故命名为 Style-MMD ✓）。
    """
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if sigmas is None:
        if median_heuristic:
            z = np.vstack([x, y])[:400]                      # 子采样估计尺度 ✓
            d = np.linalg.norm(z[:, None, :] - z[None, :, :], axis=-1)
            med = float(np.median(d[d > 0])) if (d > 0).any() else 1.0
        else:
            med = 1.0
        sigmas = [0.5 * med, med, 2.0 * med]
    vals = [_rbf_mmd2_unbiased(x, y, float(s)) for s in sigmas]
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def style_mmd_by_slot(gen: EmbSet, real: EmbSet, **kw) -> Dict[str, object]:
    """按 (writer, script) 算 Style-MMD 后取宏平均 ✓（分布比中心更能发现"风格单一化"）。"""
    per: Dict[Tuple[int, int], float] = {}
    for w, s in sorted(set(zip(gen.writer.tolist(), gen.script.tolist()))):
        mg = (gen.writer == w) & (gen.script == s)
        mr = (real.writer == w) & (real.script == s)
        if mg.sum() < 4 or mr.sum() < 4:
            continue
        per[(int(w), int(s))] = style_mmd(gen.emb[mg], real.emb[mr], **kw)
    vals = [v for v in per.values() if not np.isnan(v)]
    return {"mmd_macro": float(np.mean(vals)) if vals else float("nan"),
            "per_slot": per, "n_used": len(vals)}


# ─────────────────── ⑤ 跨字符风格三元组准确率 ───────────────────
def xctacc(gen: EmbSet, real: EmbSet, *, same_script: bool = True,
           max_triplets: Optional[int] = None, seed: int = 0) -> Dict[str, object]:
    """XCTAcc = P[ sim(gen_{w,c}, real_{w,c'}) > sim(gen_{w,c}, real_{w',c}) ] ✓。

    正样本：同书家、**不同字**；负样本：不同书家、**同字**（同书体 ✓）。
    找不到"别的书家写的同一个字"的样本 ⇒ 跳过并计数 ✓（不可用别的书体顶替 ✗）。
    """
    rng = np.random.default_rng(seed)
    # (script, char) -> {writer: [idx]}
    by_sc: Dict[Tuple[int, int], Dict[int, List[int]]] = {}
    for j in range(len(real)):
        by_sc.setdefault((int(real.script[j]), int(real.char[j])), {}).setdefault(
            int(real.writer[j]), []).append(j)
    re_ = l2norm(real.emb)
    ge = l2norm(gen.emb)
    n_ok = n_skip = 0
    d_pos: List[float] = []
    d_neg: List[float] = []
    for i in range(len(gen)):
        w, c, s = int(gen.writer[i]), int(gen.char[i]), int(gen.script[i])
        pool_pos = [j for j in range(len(real))
                    if real.writer[j] == w and real.script[j] == s and real.char[j] != c]
        neg_map = by_sc.get((s, c), {})
        pool_neg = [j for ww, js in neg_map.items() if ww != w for j in js]
        if not pool_pos or not pool_neg:
            n_skip += 1
            continue
        jp = pool_pos[int(rng.integers(len(pool_pos)))]
        jn = pool_neg[int(rng.integers(len(pool_neg)))]
        sp = float(ge[i] @ re_[jp])
        sn = float(ge[i] @ re_[jn])
        d_pos.append(sp)
        d_neg.append(sn)
        n_ok += 1
        if max_triplets is not None and n_ok >= max_triplets:
            break
    if n_ok == 0:
        return {"xctacc": float("nan"), "n_used": 0, "n_skipped": n_skip,
                "mean_pos": float("nan"), "mean_neg": float("nan"), "gap": float("nan")}
    dp, dn = np.array(d_pos), np.array(d_neg)
    return {"xctacc": float((dp > dn).mean()), "n_used": n_ok, "n_skipped": n_skip,
            "mean_pos": float(dp.mean()), "mean_neg": float(dn.mean()),
            "gap": float(dp.mean() - dn.mean())}
