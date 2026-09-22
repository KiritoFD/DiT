#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""汇总 D1（风格可见性）与 T2（风格可分性）的结果，输出对比表。

用法:
    python tools/summarize_d1_t2.py --t2-dir assets/t2_remote/assets --d1-dir assets
"""
import argparse
import glob
import io
import json
import os
import sys

# ---------------------------------------------------------------- D1

D1_ORDER = [
    ("v13_12ch_225k", "v13_12ch"),
    ("v13_base_155k", "v13_base"),
    ("v13_wd01_125k", "v13_wd01"),
    ("v14_s2_160k", "v14_s2"),
    ("v15a_150k", "v15a"),
    ("v15b_supcon_70k", "v15b_supcon"),
    ("v15c_fixed_210k", "v15c_fixed"),
]


def load_d1(d1_dir):
    """读 assets/d1_*.json，返回 {tag: metrics}。"""
    out = {}
    for f in sorted(glob.glob(os.path.join(d1_dir, "d1_*.json"))):
        try:
            d = json.load(io.open(f, encoding="utf-8"))
        except Exception:
            continue
        if isinstance(d, list):
            for r in d:
                if isinstance(r, dict) and "tag" in r:
                    out[r["tag"]] = r
        elif isinstance(d, dict):
            tag = d.get("tag") or os.path.basename(f)[3:-5]
            out[tag] = d
    return out


def d1_field(r, *names):
    for n in names:
        if n in r and r[n] is not None:
            return r[n]
    return None


def print_d1(d1):
    if not d1:
        print("  (未找到 d1_*.json —— 请把远端 assets/d1_*.json 拷到本地 assets/)")
        return

    rows = []
    for tag, r in d1.items():
        yt = d1_field(r, "ratio_y_over_t", "ratio_y_t", "y_over_t")
        dm = d1_field(r, "delta_mod_mean", "dmod", "dmod_mean")
        imb = d1_field(r, "delta_mod_layer_imbalance", "layer_imbalance")
        ax = r.get("axis_style") or {}
        dc = ax.get("dmod_rel")
        dy = ax.get("dy_emb_rel")
        deg = d1_field(r, "degenerate")
        norm_dy = dy / (yt if yt else 1.0) if dy is not None else None
        rows.append({"tag": tag, "yt": yt, "dm": dm, "imb": imb,
                     "dc": dc, "dy": dy, "ndy": norm_dy, "deg": deg})

    if not rows:
        print("  (无有效记录)")
        return

    # ---- 表 A：按 dmod（风格对 adaLN 的可见性）降序
    print("[A] 风格轴可见性（dmod 越大 = 书家条件对 adaLN 调制量影响越强）")
    print("%-18s %8s %9s %10s %9s %6s"
          % ("ckpt", "y/t", "dmod", "层不平衡", "Δy_emb", "退化"))
    print("-" * 70)
    for r in sorted(rows, key=lambda x: -(x["dm"] or -1)):
        def f(x, p="%.4f"):
            return (p % x) if isinstance(x, (int, float)) else "   -   "
        print("%-18s %8s %9s %10s %9s %6s"
              % (r["tag"], f(r["yt"]), f(r["dm"]), f(r["imb"]), f(r["dy"]),
                 ("是" if r["deg"] else "否") if r["deg"] is not None else "-"))

    # ---- 表 B：归一化 —— 风格信号占条件总能量的比例
    print()
    print("[B] 归一化：Δy_emb / ‖y_emb‖ 与 y/t 一起看（排除'幅度'解释）")
    print("%-18s %9s %9s %9s"
          % ("ckpt", "y/t", "Δy_emb", "归一化Δy"))
    print("-" * 50)
    for r in sorted(rows, key=lambda x: -(x["ndy"] or -1)):
        def f(x, p="%.4f"):
            return (p % x) if isinstance(x, (int, float)) else "   -   "
        print("%-18s %9s %9s %9s"
              % (r["tag"], f(r["yt"]), f(r["dy"]), f(r["ndy"])))


# ---------------------------------------------------------------- T2

T2_RUNS = [
    ("v13_base_50k", "v13_base_50k"),
    ("v13_12ch_post", "v13_12ch_post"),
    ("v13_wd01", "v13_wd01"),
    ("v13_styletok", "v13_styletok"),
    ("v15a_multistyle_k4", "v15a_multistyle_k4"),
    ("v15b_multistyle_k4", "v15b_multistyle_k4"),
    ("v15b_supcon", "v15b_supcon"),
    ("v15c_fixed", "v15c_fixed"),
]


def load_t2(t2_dir):
    out = {}
    for f in sorted(glob.glob(os.path.join(t2_dir, "t2_*.json"))):
        base = os.path.basename(f)[3:-5]
        key = base.rsplit("__", 1)[0]
        try:
            d = json.load(io.open(f, encoding="utf-8"))
        except Exception:
            continue
        rec = {}
        for r in d:
            tag = r.get("tag", "")
            if "生成图" in tag and "style" in tag:
                rec["gen_style"] = r
            elif "生成图" in tag and "dino" in tag:
                rec["gen_dino"] = r
            elif "GT" in tag and "style" in tag:
                rec["gt_style"] = r
            elif "GT" in tag and "dino" in tag:
                rec["gt_dino"] = r
        out[key] = rec
    return out


def ratio(a, b):
    """生成 / GT；GT 太小时返回 None（避免除零放大）。"""
    try:
        a = float(a)
        b = float(b)
    except (TypeError, ValueError):
        return None
    if abs(b) < 1e-6:
        return None
    return a / b


def print_t2(t2):
    if not t2:
        print("  (未找到 t2_*.json)")
        return

    # 天花板（所有 run 一致，取第一个能读到的）
    gt_c = gt_l = None
    gt_cd = gt_ld = None
    for rec in t2.values():
        gs = rec.get("gt_style")
        gd = rec.get("gt_dino")
        if gs:
            gt_c = gs.get("acc_centroid")
            gt_l = gs.get("acc_logreg")
        if gd:
            gt_cd = gd.get("acc_centroid")
            gt_ld = gd.get("acc_logreg")
        if gt_c is not None and gt_cd is not None:
            break
    print("  GT 天花板   手工质心 %.4f  手工线性 %.4f | DINO质心 %.4f  DINO线性 %.4f"
          % (gt_c or 0, gt_l or 0, gt_cd or 0, gt_ld or 0))
    print()
    print("%-22s %7s %7s %7s | %7s %7s %7s | %7s %7s"
          % ("run", "质心", "÷GT", "线性", "÷GT",
             "Dc质心", "÷GT", "Dc线性", "÷GT"))
    print("-" * 88)

    rows = []
    for label, key in T2_RUNS:
        rec = t2.get(key)
        if rec is None:
            # 模糊匹配
            for k, v in t2.items():
                if key in k or k in key:
                    rec = v
                    break
        if rec is None:
            continue
        gs = rec.get("gen_style") or {}
        gd = rec.get("gen_dino") or {}
        c = gs.get("acc_centroid")
        l = gs.get("acc_logreg")
        cd = gd.get("acc_centroid")
        ld = gd.get("acc_logreg")
        score = (ratio(c, gt_c) or 0) * 0.5 + (ratio(cd, gt_cd) or 0) * 0.5
        rows.append((score, label, c, l, cd, ld))

    rows.sort(key=lambda x: -x[0])
    for score, label, c, l, cd, ld in rows:
        def p(x, w=7, prec=4):
            return ("%*.4f" % (w, x)) if isinstance(x, (int, float)) else ("%*s" % (w, "-"))
        def pr(x, b):
            r = ratio(x, b)
            return ("%*.3f" % (7, r)) if r is not None else ("%*s" % (7, "-"))
        print("%-22s %s %s %s | %s %s %s | %s %s"
              % (label, p(c), pr(c, gt_c), p(l), pr(l, gt_l),
                 p(cd), pr(cd, gt_cd), p(ld), pr(ld, gt_ld)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d1-dir", default="assets/d1_remote/assets")
    ap.add_argument("--t2-dir", default="assets/t2_remote/assets")
    a = ap.parse_args()

    print("=" * 88)
    print("D1 — 风格向量对 adaLN 的可见性（真实 ckpt，forward hook 抓 t_emb/y_emb/每层 c）")
    print("=" * 88)
    print_d1(load_d1(a.d1_dir))

    print()
    print("=" * 88)
    print("T2 — 风格可分性探针（留字 CV，n=249 / 41 书家，随机基线 1/41=0.0244）")
    print("=" * 88)
    print_t2(load_t2(a.t2_dir))


if __name__ == "__main__":
    main()
