#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""汇总 D3 空间风格探针结果。

D3 判据（本轮修正版）：
  - style_D        : 固定字+固定噪声+固定骨架，只换书家 -> 像素 L1 差异
  - noise_D        : 固定字+固定书家+固定骨架，只换噪声 -> 像素 L1 差异（噪声地板）
  - ratio          = style_D / noise_D
  - lb (下界)      = (style_D - noise_std) / noise_D      <-- 严格判据
  - in/bg          = mean(D|笔画内) / mean(D|背景)         <-- 注意：噪声地板同样 ≈2.5，无区分力

判读：
  1) lb <= 0  => 换书家的差异完全落在噪声波动内 => 风格没进像素
  2) in/bg 高  且 噪声地板 in/bg 同样高       => 骨架约束的必然结果，不能作为风格证据
"""
import json
import glob
import os
import sys


def load(f):
    d = json.load(open(f, encoding="utf-8"))
    s = d.get("summary", d)
    n = os.path.basename(f).replace("d3_", "").replace(".json", "")
    return n, s


def main():
    pat = sys.argv[1] if len(sys.argv) > 1 else "_sync_work/d3_out/*.json"
    rows = [load(f) for f in sorted(glob.glob(pat))]
    if not rows:
        print("no json found:", pat)
        return

    hdr = ("%-30s %8s %7s %7s %7s %7s %8s %8s"
           % ("run", "s/n_lb", "s/n_g", "S_ink", "N_ink", "in/bg", "Nratio", "sig/D"))
    print(hdr)
    print("-" * len(hdr))

    verdicts = []
    for n, s in rows:
        sd = s.get("style_D_global", float("nan"))
        nd = s.get("noise_D_global", float("nan"))
        sig = s.get("noise_D_global_std", float("nan"))
        lb = s.get("style_over_noise_global_lb", float("nan"))
        sg = s.get("style_over_noise_global", float("nan"))
        s_ink = s.get("style_D_in_ink", float("nan"))
        n_ink = s.get("noise_D_in_ink", float("nan"))
        sr = s.get("style_ratio", float("nan"))
        nr = s.get("noise_ratio", float("nan"))
        pairs = s.get("n_pairs", "?")

        print("%-30s %8.3f %7.3f %7.3f %7.3f %7.2f %8.2f %8.2f"
              % (n, lb, sg, s_ink, n_ink, sr, nr, sig / max(sd, 1e-9)))

        v = []
        v.append("pairs=%s" % pairs)
        if lb <= 0:
            v.append("FAIL: 换书家差异 < 噪声 -> 风格未进像素")
        elif lb < 0.3:
            v.append("WEAK: 风格信号微弱 (lb=%.2f)" % lb)
        else:
            v.append("PASS: lb=%.2f" % lb)
        if nr >= 1.5:
            v.append("(注: 噪声地板 in/bg=%.2f >=1.5, 故 in/bg 不可作风格证据)" % nr)
        verdicts.append((n, " | ".join(v)))

    print()
    print("=== 逐 run 判读 ===")
    for n, v in verdicts:
        print("  %-30s %s" % (n, v))

    # 纯书家对照（script-mode=same）
    same = [(n, s) for n, s in rows if "_same_same" in n]
    if same:
        print()
        print("=== script-mode=same 纯书家对照（已消掉书体混杂）===")
        for n, s in same:
            sd = s.get("style_D_global", float("nan"))
            nd = s.get("noise_D_global", float("nan"))
            print("  %-30s style_D=%.4f  noise_D=%.4f  ratio=%.3f  pairs=%s"
                  % (n, sd, nd, sd / max(nd, 1e-9), s.get("n_pairs", "?")))
        print("  -> 若与 script-mode=any 的 ratio 相近，说明'书体'并没有贡献额外差异")


if __name__ == "__main__":
    main()
