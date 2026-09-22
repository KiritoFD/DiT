# -*- coding: utf-8 -*-
"""T1 — 笔画级统计量的「组间 / 组内」方差比。

⚠⚠ 重要局限（2026-09-22 实测踩到，必须知道）⚠⚠

本工具在 **strict 评测集上无效**，原因有三：

1. **strict 集是"未见三元组"** —— 每个字只对应 **1 位书家**
   （实测：249 行 / 242 字，其中同字 >= 2 书家的只有 **6 个，2.5%**）。
   于是「字内书家比较」没有数据，`f_ratio_within_char` 返回 0。

2. **朴素单因素 `f_ratio` 会被"字形"严重污染** —— 它的"书家组间方差"里
   混进了"字与字之间的差异"。实测 `bbox_h`（包围盒高度）常年排第一，
   但那是**字形**不是**笔法**。训练集上朴素 F 能到 12.9，字内版却是 0。

3. 即使数据够（训练集有 73.4% 的字满足同字多书家），
   每个 (字, 书家) 组合**只有 1 个样本**时组内自由度 = 0，方差无法估计。
   此时「字内书家判别力」**退化成 T2 的留字 CV** —— 两者本质是同一个问题。

→ **结论：ckpt 的风格评测请用 T2（留字 CV + GT 天花板）或 D3（固定字换书家）。
   本工具只保留用于「特征本身是否有效」的检验，不要拿它的数字判 ckpt。**

它仍能回答一个有用的问题：**手工笔画特征里哪些维度真的携带书家信息**
（在训练集的「同字多书家」子集上跑，看逐维 F）—— 这关系到 T2 该不该换特征。

算法:
    对每个特征维度 d，按「书家」分组:
        between_d = 组均值的方差（跨书家）
        within_d  = 组内方差的平均（同书家内）
        F_d       = between_d / within_d          （越大 = 该维度越能区分书家）
    同时算 GT 的同一指标作为天花板 + 置换检验给经验零分布。

用法:
    python tools/probe_style_variance.py \
        --samples assets/ink_eval/v13_base_50k__0155000__strict \
        --eval-csv assets/eval_v13_strict.csv \
        --out assets/t1_v13_base_strict.json
"""
import argparse
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FEAT_NAMES = [
    "ink_ratio", "gray_p10", "gray_p25", "gray_p50", "gray_p75", "gray_p90",
    "gray_std", "stroke_w_mean", "stroke_w_med", "stroke_w_std", "stroke_w_p90",
    "bbox_h", "bbox_w", "bbox_ar", "centroid_dx", "centroid_dy", "edge_sharp",
    "run_h", "run_v",
    "dens_00", "dens_01", "dens_02", "dens_10", "dens_11", "dens_12",
    "dens_20", "dens_21", "dens_22", "erode_ratio",
]
assert len(FEAT_NAMES) == 29, len(FEAT_NAMES)


def _collect(d, prefix):
    """按数字序收集 <prefix><i>.png —— 与 T2 同口径（锚定结尾）。"""
    pat = re.compile(r"^%s(\d+)\.png$" % re.escape(prefix))
    items = []
    for f in os.listdir(d):
        m = pat.match(f)
        if m:
            items.append((int(m.group(1)), os.path.join(d, f)))
    items.sort()
    return [p for _, p in items]


def load_groups(eval_csv, n):
    """从 eval csv 取 (calligrapher, character)，行序与图片数字序对应。"""
    rows = list(csv.DictReader(open(eval_csv, encoding="utf-8")))[:n]
    calligs, chars = [], []
    for r in rows:
        calligs.append(str(r.get("calligrapher") or r.get("callig") or "?"))
        chars.append(str(r.get("character") or r.get("char") or "?"))
    return calligs, chars


def f_ratio(X, groups):
    """逐维度 + 总体的 between/within 方差比（单因素：按 groups 分组）。

    ⚠ 这个版本会被**混杂因素**污染：如果每个组里混了多种"字"，
    则"组间方差"里含了字的差异。做风格判定请用 `f_ratio_within_char`。

    X: (n, d) float64 ndarray
    groups: list[str] 长度 n
    返回 (per_dim F, overall F)
    """
    import numpy as np
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    uniq = sorted(set(groups))
    between = np.zeros(d)
    within = np.zeros(d)
    df_b, df_w = 0, 0
    for g in uniq:
        idx = [i for i, x in enumerate(groups) if x == g]
        if len(idx) < 2:
            continue
        sub = X[idx]
        gm = sub.mean(axis=0)
        between += len(idx) * (gm - X.mean(axis=0)) ** 2
        within += ((sub - gm) ** 2).sum(axis=0)
        df_b += 1
        df_w += len(idx) - 1
    if df_b == 0 or df_w == 0:
        return np.zeros(d), 0.0
    F = (between / df_b) / np.maximum(within / df_w, 1e-12)
    overall = (between.sum() / df_b) / max(within.sum() / df_w, 1e-12)
    return F, float(overall)


def f_ratio_within_char(X, calligs, chars):
    """★ 关键修正版：**在每个字内部**比较书家 —— 消掉"字形"混杂。

    做法（配对/分层）：对每个字 c，只取同时有 >= 2 位书家的那些行，
        between_c = 该书家均值的方差（跨书家，**字内**）
        within_c  = 组内方差
    然后把所有字的分母/分子各自累加。

    这样 F 完全不吃"字不同"的信息：同字内部书家之间的笔画差异才是风格。
    与 T2 的 `group_by=char`（留字 CV）是同一个纪律。

    X: (n, d)；calligs / chars: 长度 n 的标签
    返回 (per_dim F, overall F, 用了多少个字/多少行)
    """
    import numpy as np
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    from collections import defaultdict
    by_char = defaultdict(list)
    for i, c in enumerate(chars):
        by_char[c].append(i)

    between = np.zeros(d)
    within = np.zeros(d)
    df_b, df_w = 0, 0
    n_used_rows, n_used_chars = 0, 0
    for c, idxs in by_char.items():
        # 该字下的书家 -> 行
        g2 = defaultdict(list)
        for i in idxs:
            g2[calligs[i]].append(i)
        g2 = {k: v for k, v in g2.items() if len(v) >= 1}
        if len(g2) < 2:
            continue
        sub_all = X[[i for v in g2.values() for i in v]]
        gm_all = sub_all.mean(axis=0)
        for k, v in g2.items():
            s = X[v]
            gm = s.mean(axis=0)
            between += len(v) * (gm - gm_all) ** 2
            within += ((s - gm) ** 2).sum(axis=0)
            df_b += 1
            df_w += len(v) - 1
            n_used_rows += len(v)
        n_used_chars += 1
    if df_b == 0 or df_w == 0:
        return np.zeros(d), 0.0, 0, 0
    F = (between / df_b) / np.maximum(within / df_w, 1e-12)
    overall = (between.sum() / df_b) / max(within.sum() / df_w, 1e-12)
    return F, float(overall), n_used_chars, n_used_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--eval-csv", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--n-perm", type=int, default=20,
                    help="置换检验次数（打乱书家标签，给 F 的经验零分布）")
    ap.add_argument("--max-feats", type=int, default=0,
                    help=">0 时只取前 N 个特征（调试用）")
    a = ap.parse_args()

    import numpy as np

    gen = _collect(a.samples, "g")
    gts = _collect(a.samples, "gt")
    if not gen:
        print("目录里没找到 g<i>.png：%s" % a.samples)
        sys.exit(1)
    n = min(len(gen), len(gts)) if gts else len(gen)
    print("[T1] %s" % a.samples)
    print("     生成图 %d 张, GT %d 张, 用 %d" % (len(gen), len(gts), n))

    calligs, chars = load_groups(a.eval_csv, n)
    n = min(n, len(calligs))
    gen, gts, calligs, chars = gen[:n], gts[:n], calligs[:n], chars[:n]
    print("     书家 %d 个, 字 %d 个 (来自 %s)"
          % (len(set(calligs)), len(set(chars)), os.path.basename(a.eval_csv)))

    from tools.probe_style_separability import feat_style

    results = {}
    for tag, paths in (("生成图", gen), ("GT", gts)):
        if not paths:
            continue
        X = np.asarray(feat_style(paths), dtype=np.float64)
        if a.max_feats > 0:
            X = X[:, :a.max_feats]
        # ★ 主指标：**字内**书家方差比（消掉字形混杂）
        Fw, overall_w, nc, nr = f_ratio_within_char(X, calligs, chars)
        # 参考：朴素单因素（会被字形污染，只作对照）
        F, overall = f_ratio(X, calligs)
        rng = np.random.RandomState(0)
        nulls = []
        for _ in range(a.n_perm):
            perm = list(calligs)
            rng.shuffle(perm)
            nulls.append(f_ratio_within_char(X, perm, chars)[1])
        null_m = float(np.mean(nulls))
        null_s = float(np.std(nulls))
        z = (overall_w - null_m) / null_s if null_s > 1e-12 else float("nan")
        results[tag] = {"overall_F": overall_w, "F_within_char": overall_w,
                        "overall_F_naive": overall,
                        "null_mean": null_m, "null_std": null_s, "z": z,
                        "per_dim": Fw.tolist(), "per_dim_naive": F.tolist(),
                        "n": int(X.shape[0]), "d": int(X.shape[1]),
                        "n_chars_used": nc, "n_rows_used": nr}
        print("   %-8s [字内] F = %8.4f   置换零分布 %.4f ± %.4f   z = %+.1f"
              % (tag, overall_w, null_m, null_s, z))
        print("            (对照 朴素单因素 F = %.4f —— 含字形混杂，偏乐观)"
              % overall)

    if "生成图" in results and "GT" in results:
        r = results["生成图"]["overall_F"] / max(results["GT"]["overall_F"], 1e-12)
        print()
        print("   → 生成 / GT 的**字内** F 比 = %.3f   %s"
              % (r, "✅ 信号在" if r >= 0.5 else "✗ 信号弱"))
        results["ratio_gen_over_gt"] = r

    # ---- 逐维度排行（用生成图）
    key = "生成图" if "生成图" in results else list(results)[0]
    F = np.asarray(results[key]["per_dim"])
    order = np.argsort(-F)
    print()
    print("   逐维度 F（书家区分度）Top-8 —— 什么笔画量在携带风格：")
    for i in order[:8]:
        nm = FEAT_NAMES[i] if i < len(FEAT_NAMES) else "feat%d" % i
        print("      %-16s F = %8.4f" % (nm, F[i]))
    print("   最弱 Bottom-4：")
    for i in order[-4:]:
        nm = FEAT_NAMES[i] if i < len(FEAT_NAMES) else "feat%d" % i
        print("      %-16s F = %8.4f" % (nm, F[i]))
    results["feat_names"] = FEAT_NAMES

    # ---- ⚠ 关键对照：把「几何/字形」维度剔除后重算
    # bbox_* 是包围盒（字形尺寸），centroid_* 是墨心偏移（也受字形影响），
    # dens_* 是固定网格密度（粗糙，同样吃字形）。这些高 F 可能只反映"字不同"，
    # 而不是"书家不同"。剥掉它们再看颜色/笔画维度，才是纯笔法风格的上限。
    GEO = {"bbox_h", "bbox_w", "bbox_ar", "centroid_dx", "centroid_dy",
           "dens_00", "dens_01", "dens_02", "dens_10", "dens_11", "dens_12",
           "dens_20", "dens_21", "dens_22"}
    keep_idx = [i for i, nm in enumerate(FEAT_NAMES) if nm not in GEO]
    print()
    print("   [对照] 剔除几何/字形维度（bbox_* / centroid_* / dens_*）后重算")
    print("          保留 %d 维：%s"
          % (len(keep_idx), ", ".join(FEAT_NAMES[i] for i in keep_idx)))
    for tag, paths in (("生成图", gen), ("GT", gts)):
        if not paths:
            continue
        Xa = np.asarray(feat_style(paths), dtype=np.float64)[:, keep_idx]
        Fa, ova, _nc, _nr = f_ratio_within_char(Xa, calligs, chars)
        rng = np.random.RandomState(0)
        nulls = []
        for _ in range(a.n_perm):
            perm = list(calligs)
            rng.shuffle(perm)
            nulls.append(f_ratio_within_char(Xa, perm, chars)[1])
        nm_ = float(np.mean(nulls))
        ns_ = float(np.std(nulls))
        print("          %-8s [字内] F = %8.4f   零分布 %.4f ± %.4f   z = %+.1f"
              % (tag, ova, nm_, ns_,
                 (ova - nm_) / ns_ if ns_ > 1e-12 else float("nan")))
        results.setdefault("nongeo", {})[tag] = {
            "overall_F": ova, "null_mean": nm_, "null_std": ns_,
            "per_dim": Fa.tolist(), "keep_idx": keep_idx}
    if "nongeo" in results and len(results["nongeo"]) == 2:
        r2 = (results["nongeo"]["生成图"]["overall_F"]
              / max(results["nongeo"]["GT"]["overall_F"], 1e-12))
        print("          → 生成 / GT = %.3f   %s"
              % (r2, "✅ 纯笔法信号在" if r2 >= 0.5 else "✗ 纯笔法信号弱"))
        results["nongeo"]["ratio_gen_over_gt"] = r2

    if a.out:
        json.dump(results, open(a.out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print("\n-> %s" % a.out)


if __name__ == "__main__":
    main()
