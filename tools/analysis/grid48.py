#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""grid48.py — 48 上 4-arm 消融的 2x2 网格（aug x C2OT）匹配步数主效应

网格四格（grid = aug × C2OT）:
  (aug,   C2OT) = v68            <- 历史主模型
  (aug,   noOT) = sym3x_noOT
  (noaug, C2OT) = noaug_c2ot
  (noaug, noOT) = noaug_noOT
注意: sym3x_naiveOT 是**第 5 臂**(朴素 OT), 不在网格内(驱动脚本已注明主动放弃)。
"""
import glob
import json
import os
import sys

from collections import defaultdict

CELLS = {
    ("aug", "C2OT"): ["assets/results/ablation/v68*", "assets/runs/v68*", "assets/results/v68*"],
    ("aug", "noOT"): ["assets/results/ablation/sym3x_noOT/*"],
    ("noaug", "C2OT"): ["assets/results/ablation/noaug_c2ot/*"],
    ("noaug", "noOT"): ["assets/results/ablation/noaug_noOT/*"],
}
MET = ("ssim", "lpips", "frag", "ink_iou")


def load(pats):
    d = defaultdict(dict)
    for pat in pats:
        for p in glob.glob(pat + "/checkpoints/eval_auto_*.json"):
            try:
                j = json.load(open(p, encoding="utf-8"))
            except Exception:
                continue
            if isinstance(j, dict) and j.get("set") in ("eval200fix", "eval200_fixed"):
                d[j["step"]] = {k: j.get(k) for k in MET}
    return d


def main():
    cur = {}
    for c, pats in CELLS.items():
        d = load(pats)
        cur[c] = d
        ks = sorted(d)
        print(f"  {str(c):<18} 评测点: {ks if ks else '（无）'}")
    common = None
    for c, d in cur.items():
        s = set(d)
        common = s if common is None else (common & s)
    common = sorted(common or [])
    print(f"\n  四格共同评测点: {common}")
    if not common:
        return
    for step in [s for s in common if s in (20000, 40000, 100000)] or common[-3:]:
        print(f"\n  === step {step} ===")
        print(f"  {'格 (aug,OT)':<18}{'SSIM':>9}{'LPIPS':>9}{'frag':>9}{'ink':>9}")
        for c in (("aug", "C2OT"), ("aug", "noOT"), ("noaug", "C2OT"), ("noaug", "noOT")):
            m = cur[c].get(step)
            if m:
                print(f"  {str(c):<18}{m['ssim']:>9.4f}{m['lpips']:>9.4f}{(m['frag'] or 0):>9.3f}{(m['ink_iou'] or 0):>9.4f}")
            else:
                print(f"  {str(c):<18}{'—':>9}")
        # 主效应: aug 在 C2OT 下 / C2OT 在 aug 下
        try:
            a_c = cur[("aug", "C2OT")][step]; a_n = cur[("aug", "noOT")][step]
            n_c = cur[("noaug", "C2OT")][step]; n_n = cur[("noaug", "noOT")][step]
            print(f"  aug 效应 @C2OT  : dSSIM {a_c['ssim']-n_c['ssim']:+.4f}  dLPIPS {a_c['lpips']-n_c['lpips']:+.4f}  "
                  f"dfrag {(a_c['frag'] or 0)-(n_c['frag'] or 0):+.3f}")
            print(f"  aug 效应 @noOT  : dSSIM {a_n['ssim']-n_n['ssim']:+.4f}  dLPIPS {a_n['lpips']-n_n['lpips']:+.4f}  "
                  f"dfrag {(a_n['frag'] or 0)-(n_n['frag'] or 0):+.3f}")
            print(f"  C2OT 效应 @aug : dSSIM {a_c['ssim']-a_n['ssim']:+.4f}  dLPIPS {a_c['lpips']-a_n['lpips']:+.4f}  "
                  f"dfrag {(a_c['frag'] or 0)-(a_n['frag'] or 0):+.3f}")
            print(f"  C2OT 效应 @noaug: dSSIM {n_c['ssim']-n_n['ssim']:+.4f}  dLPIPS {n_c['lpips']-n_n['lpips']:+.4f}  "
                  f"dfrag {(n_c['frag'] or 0)-(n_n['frag'] or 0):+.3f}")
        except KeyError:
            print("  （四格不齐, 跳过主效应）")


if __name__ == "__main__":
    main()
