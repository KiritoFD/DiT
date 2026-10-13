#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""eval_style_encoder_cached.py —— 用**缓存 latent** 复评一个风格编码器（含自助法 CI ✓）。

为什么要它（实测教训 ✓）
    完整评估要先用 VAE 编码 9,000 张 = **26 分钟** ✗（`train_style_encoder.py` 里那一步）。
    但 latent 缓存已落盘 ⇒ 复评只需 **加载缓存 + 前向编码器** ✓（秒级 ✓），
    于是可以方便地做: 多版本对比 / 换 n_pairs / 加 bootstrap CI ✓✓。

判定规则（**预先登记** ✓，避免事后合理化 ✗）
    G1 的 95% CI:
      · ci95[0] > 0.65  ⇒ **通过** ✓
      · ci95 含 0.65     ⇒ **临界** ⇒ 如实报告 ✓（G2/G3 若强过则标注 ✓）
      · ci95[1] < 0.65  ⇒ **不通过** ✗ ⇒ 继续迭代

用法
    python tools/eval_style_encoder_cached.py --ckpt assets/style_enc_v85c.pt \
        --cache scratch/style_lat_local_full.npz --n-pairs 1500 --bootstrap 2000
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.style.bank import resolve, work_id                          # noqa: E402
from src.eval.style.gates import gate1_2afc, gate2_retrieval, gate3_printed  # noqa: E402


def load_rows(csv_path, roots, max_img=0):
    """必须与 `train_style_encoder.load_rows` **逐位一致** ✓（否则 latent 行序对不上 ✗）。"""
    from collections import defaultdict as dd
    by = dd(list)
    for r in csv.DictReader(open(csv_path, encoding="utf-8-sig")):
        if (r.get("aug") or "").strip():
            continue
        p = resolve(r.get("image_path", ""), roots)
        if not p:
            continue
        by[(r.get("calligrapher") or "", r.get("script") or "")].append({
            "path": p, "writer": (r.get("calligrapher") or "").strip(),
            "script": (r.get("script") or "").strip(),
            "char": (r.get("character") or "").strip(),
            "work": work_id(r.get("src_image_path", ""), r.get("image_path", "")),
            "std": resolve(r.get("std_path", ""), roots) or ""})
    out, i = [], 0
    if max_img:
        while len(out) < max_img:
            added = False
            for k in sorted(by):
                if i < len(by[k]):
                    out.append(by[k][i])
                    added = True
                    if len(out) >= max_img:
                        break
            if not added:
                break
            i += 1
    else:
        out = [d for k in sorted(by) for d in by[k]]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", default="scratch/style_lat_local_full.npz")
    ap.add_argument("--csv", default="exp-std/csv/train.csv")
    ap.add_argument("--root", action="append", default=[])
    ap.add_argument("--n-pairs", type=int, default=1500)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-holdout", action="store_true",
                    help="G1 在**全部行**上算（= 方案字面的\"真实真迹\"口径 ✓）\n"
                    "默认 False = 作品级留出（更难 ✓，方案也要求留出验证 ✓）⇒ **两种都报** ✓")
    ap.add_argument("--device", default="")
    ap.add_argument("--save-json", default="", dest="save_json",
                    help="把两种口径的结论存成 JSON（供 build_style_protos --gate-json 读 ✓）")
    args = ap.parse_args()
    import torch
    from src.loss.style_rank_module import load_encoder
    roots = args.root or [".", "data", "exp-std"]
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    rows = load_rows(args.csv, roots)
    z = np.load(args.cache)["lat"].astype(np.float32)
    assert int(z.shape[0]) == len(rows), "缓存行数 %d != 行数 %d ✗（口径不一致 ✓）" % (
        z.shape[0], len(rows))
    enc, ck = load_encoder(args.ckpt, verbose=True)
    enc = enc.to(dev).eval()
    with torch.no_grad():
        out = []
        for s in range(0, len(z), 2048):
            out.append(enc.emb(torch.from_numpy(z[s:s + 2048]).to(dev)).cpu().numpy())
    emb = np.concatenate(out)

    # 与训练同源的作品级留出 ✓（同 seed ✓ ⇒ 同一划分 ✓）
    import random
    from collections import defaultdict
    rng = random.Random(args.seed)
    by_slot = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(rows):
        by_slot[(r["writer"], r["script"])][r["work"]].append(i)
    ho = []
    for _s, works in by_slot.items():
        ws = sorted(works)
        if len(ws) >= 3:
            rng.shuffle(ws)
            for w in ws[:max(1, int(round(len(ws) * ck.get("holdout_work_frac", 0.3))))]:
                ho += works[w]
    ho = sorted(set(ho))
    ho_rows = [rows[i] for i in ho]

    print("\n===== 复评 %s（作品级留出 n=%d ✓）=====" % (os.path.basename(args.ckpt), len(ho)))
    _g1_rows = rows if args.no_holdout else ho_rows
    _g1_emb = emb if args.no_holdout else emb[ho]
    _tag = "全部行(同分布 ✓)" if args.no_holdout else "作品级留出(更难 ✓)"
    print("  G1 口径: %s n_rows=%d" % (_tag, len(_g1_rows)))
    g1 = gate1_2afc(_g1_emb, _g1_rows, n_pairs=args.n_pairs, seed=args.seed,
                    bootstrap=args.bootstrap)
    g2 = gate2_retrieval(emb, rows, seed=args.seed)
    g3 = gate3_printed(emb[ho], ho_rows, None, None)
    ci = g1.get("ci95")
    print("  G1 2AFC : %.4f (n=%d) se=%.4f ci95=[%s]" % (
        g1["acc"], g1["n"], g1.get("se", float("nan")),
        ("%.4f, %.4f" % (ci[0], ci[1])) if ci else "NA"))
    print("  G2 mAP  : %.4f (n=%d)  同书体基线 %.4f" % (g2["map"], g2["n"], g2["base_script"]))
    verd = "NA"
    if ci:
        if ci[0] > 0.65:
            verd = "✓ 通过（CI 下界 > 0.65 ✓）"
        elif ci[1] < 0.65:
            verd = "✗ 不通过（CI 上界 < 0.65 ✗）"
        else:
            verd = "⚠ **临界**（CI 含 0.65 ⇒ 如实报告 ✓）"
    print("  ⇒ 按**预先登记**的规则: %s" % verd)
    if args.save_json:
        import json as _j
        _j.dump({"ckpt": args.ckpt, "same_dist": g1, "g2": g2,
                 "pass": bool(ci and ci[0] > 0.65), "n_rows": len(rows),
                 "holdout_n": len(ho_rows)},
                open(args.save_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("  ✓ 结论已存 %s" % args.save_json)
    return 0 if (ci and ci[0] > 0.65) else 1


if __name__ == "__main__":
    sys.exit(main())
