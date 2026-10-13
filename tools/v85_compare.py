#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v85_compare.py —— 按方案 §7 出**双轴对照表**（v84A / v85A / Control / v85B / v85C ✓）。

方案 §7 的成功判据（逐条落实 ✓）
    | Writer XCTAcc        | 明显超随机; DINO 指标不恶化 |
    | Writer Prototype Margin | 较 v84A 显著提高 |
    | BSD-width ↓          | ≤1.5, 争取更低 |
    | Ink IoU ↑            | 初轮不低于 ~0.27 |
    | LPIPS-Alex ↓         | 保持或改善 |
    | OCR                  | 按频次分层 |
    | near-blank ↓         | 先争取 <5% |
    | 人工风格偏好           | 缺同协议对照 |

本工具做**可自动判定**的部分 ✓：
    · 风格轴: 调 tools/style_eval.py（DINO-S/CLIP-S + mAP + XCTAcc + HWD-CN + Style-MMD + BSD ✓）
    · 内容轴: 直接读各 run 的 `eval_stdskel_summary.csv`（取 `*_mean` 列 ✓）
并打印成**并排表** ✓（含 v84A 基线与 §7 目标 ✓），人工项留空 ✓。

用法
    python tools/v85_compare.py --set v84A@200k=assets/style_eval_src/step0200000 \
        --set v85A@30k=assets/style_eval_src/v85A_step0030000 \
        --set control@5k=assets/style_eval_src/ctrl_step0005000 \
        --csv-map v85A@30k=assets/results/v85A_style_ada/*/eval_stdskel_summary.csv \
        --out assets/style_eval_local/v85_compare
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = {"ink_iou": 0.2840, "lpips": 0.3551, "ssim": 0.5846, "bsd_width": None,
        "dino_s": 0.7771, "clip_s": 0.8386, "xctacc": None}
TARGET = {"bsd_width_max": 1.5, "ink_iou_min": 0.27, "near_blank_max": 0.05}


def read_csv_metrics(pat, setname="eval200fix"):
    fs = sorted(glob.glob(pat))
    if not fs:
        return {}
    rows = list(csv.DictReader(open(fs[-1], encoding="utf-8")))
    rows = [r for r in rows if (r.get("set") or "") == setname] or rows
    if not rows:
        return {}
    def g(r, *ks):
        for k in ks:
            try:
                return float(r[k])
            except Exception:
                pass
        return float("nan")
    last = rows[-1]
    return {"step": last.get("step"), "ssim": g(last, "ssim_mean"),
            "ink_iou": g(last, "ink_iou_mean"), "lpips": g(last, "lpips_mean"),
            "mse": g(last, "mse_mean"), "near_blank": g(last, "near_blank_rate"),
            "frag": g(last, "frag_ratio"), "hole": g(last, "hole_pred")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", action="append", default=[], help="name=dir（生成样本目录 ✓）")
    ap.add_argument("--csv-map", action="append", default=[],
                    help="name=glob（该 run 的 summary CSV ✓ 供内容轴 ✓）")
    ap.add_argument("--out", default="assets/style_eval_local/v85_compare")
    ap.add_argument("--max-ref", default="3000")
    ap.add_argument("--bsd-ref-max", default="1000")
    a = ap.parse_args()
    sets = [s.split("=", 1) for s in a.set if "=" in s]
    cmap = dict(s.split("=", 1) for s in a.csv_map if "=" in s)
    if not sets:
        print("需要 --set name=dir")
        return 2
    os.makedirs(a.out, exist_ok=True)
    args = []
    for n, d in sets:
        args += ["--gen", "%s=%s" % (n, d)]

    reports = {}
    for enc, nm in (("dinov2_vitl14", "DINO-S"), ("clip_vitl14", "CLIP-S")):
        cmd = [sys.executable, "tools/style_eval.py", "--enc", enc, "--pool", "cls",
               "--csv", "exp-std/csv/train.csv", "--root", ".", "--root", "data",
               "--gen-map", "exp-std/csv/eval200_fixed.csv", *args,
               "--out", os.path.join(a.out, nm), "--max-ref", a.max_ref,
               "--baselines", "1", "--bsd", "1", "--bsd-ref-max", a.bsd_ref_max,
               "--batch", "24"]
        print("=== 风格轴 %s ===" % nm, flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        p = os.path.join(a.out, nm, "style_report.json")
        reports[nm] = json.load(open(p, encoding="utf-8")) if os.path.isfile(p) else None
        if reports[nm] is None:
            print("  ✗ %s" % (r.stderr or "")[-160:])

    print("\n===== 方案 §7 双轴对照（v84A@200k 为基线 ✓）=====")
    hdr = ("%-22s | %8s %8s | %7s %7s %8s | %8s %8s %8s %8s %9s"
           % ("集合", "DINO-S", "CLIP-S", "mAP", "XCT", "HWD-CN", "ssim", "ink_iou",
              "lpips", "near_blank", "BSD宽"))
    print(hdr)
    print("-" * len(hdr))
    for n, _d in sets:
        row = [n]
        for nm in ("DINO-S", "CLIP-S"):
            g = ((reports.get(nm) or {}).get("sets") or {}).get(n) or {}
            row.append((g.get("style_sim") or {}).get("s_all", float("nan")))
        g = ((reports.get("DINO-S") or {}).get("sets") or {}).get(n) or {}
        row += [g.get("retrieval", {}).get("map", float("nan")),
                g.get("xctacc", {}).get("xctacc", float("nan")),
                g.get("hwd_cn", {}).get("macro", float("nan"))]
        cm = read_csv_metrics(cmap[n]) if n in cmap else {}
        row += [cm.get("ssim", float("nan")), cm.get("ink_iou", float("nan")),
                cm.get("lpips", float("nan")), cm.get("near_blank", float("nan")),
                (g.get("bsd") or {}).get("width", float("nan"))]
        print("%-22s | %8.4f %8.4f | %7.4f %7.4f %8.3f | %8.4f %8.4f %8.4f %8.4f %9.4f"
              % tuple([row[0]] + [float(x) for x in row[1:]]))
    print("\n  §7 目标: BSD宽 ≤%.2f · ink_iou ≥%.2f · near_blank <%.0f%% · "
          "XCTAcc/margin 提高 · DINO 不恶化" % (TARGET["bsd_width_max"],
          TARGET["ink_iou_min"], 100 * TARGET["near_blank_max"]))
    print("  基线 v84A@200k: ssim %.4f · ink_iou %.4f · lpips %.4f · DINO-S %.4f · CLIP-S %.4f"
          % (BASE["ssim"], BASE["ink_iou"], BASE["lpips"], BASE["dino_s"], BASE["clip_s"]))
    json.dump({"sets": [n for n, _ in sets], "reports": reports}, 
              open(os.path.join(a.out, "v85_compare.json"), "w", encoding="utf-8"),
              ensure_ascii=False, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())
