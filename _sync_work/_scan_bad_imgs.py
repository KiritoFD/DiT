# -*- coding: utf-8 -*-
"""_scan_bad_imgs.py — 全库扫描"全黑噪声"坏图, 量化影响范围.

判据: 正常书法图 = **白底**(全局灰度均值高) + 笔画(前景占比中低).
      坏图 (实测 unicalli_chars/970622.png 等) = 整幅暗 + 前景占比极高, 且**没有字形**.
      用 (灰度均值 < 120 且 前景占比 > 0.5) 作为保守的坏图判据。
"""
import csv
import os
import sys
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CSV = "assets/train_base_noaug.csv"      # 只扫原图 (增强图继承原图的问题)


def stat(p):
    a = np.asarray(Image.open(p).convert("L"), dtype=np.uint8)
    return float(a.mean()), float((a < 127).mean())


rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
print(f"[scan] {CSV}: {len(rows)} 张原图 ...", flush=True)

bad, ok = [], 0
by_src_bad, by_src_all = Counter(), Counter()
for i, r in enumerate(rows):
    p = r.get("image_path", "")
    src = p.split("/")[2] if len(p.split("/")) > 2 else "?"
    by_src_all[src] += 1
    if not os.path.exists(p):
        continue
    try:
        m, fg = stat(p)
    except Exception:
        bad.append((p, -1, -1, src, r.get("character", "")))
        by_src_bad[src] += 1
        continue
    if m < 120 and fg > 0.5:
        bad.append((p, m, fg, src, r.get("character", "")))
        by_src_bad[src] += 1
    else:
        ok += 1
    if (i + 1) % 10000 == 0:
        print(f"  {i+1}/{len(rows)}  坏图累计={len(bad)}", flush=True)

print(f"\n=== 结果 ===")
print(f"  正常={ok}  坏图={len(bad)}  ({100*len(bad)/max(len(rows),1):.2f}%)")
print(f"\n  按数据源:")
for s, n in by_src_all.most_common():
    b = by_src_bad.get(s, 0)
    print(f"    {s:24s} 总={n:6d} 坏={b:5d} ({100*b/max(n,1):5.2f}%)")
print(f"\n  坏图样例 (path, 灰度均值, 前景占比, 字):")
for p, m, fg, s, ch in bad[:15]:
    print(f"    {p}  mean={m:6.1f} fg={fg:.3f} char={ch}")

# 受影响的唯一字符数
ch_bad = Counter(b[4] for b in bad if b[4])
print(f"\n  受影响的唯一字符数={len(ch_bad)}  样例={ch_bad.most_common(10)}")
with open("/tmp/bad_imgs.txt", "w", encoding="utf-8") as f:
    for p, m, fg, s, ch in bad:
        f.write(f"{p}\t{m:.1f}\t{fg:.3f}\t{ch}\n")
print("  -> /tmp/bad_imgs.txt")
