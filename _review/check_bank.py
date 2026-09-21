"""查 gradio 的"库存"到底是什么，以及为什么 '阜'(楷) 不在里面。"""
import os
import re

import numpy as np

os.chdir("/root/Workspace/xy/DiT")

print("=== gradio 里与库存/可选字有关的代码 ===")
src = open("gradio_stdskel.py", encoding="utf-8").read()
for i, line in enumerate(src.splitlines(), 1):
    if re.search(r"库存|bank|不在|not in|only|available|候选|choices", line):
        print(f"  {i}: {line.strip()[:120]}")

print()
print("=== bank npz ===")
P = "_sync_work/data/skel/skel_bank_std1_v8.npz"
if not os.path.exists(P):
    print("  不存在:", P)
else:
    print(f"  {P}  {os.path.getsize(P)/1e6:.1f} MB")
    z = np.load(P, allow_pickle=True)
    print(f"  keys({len(z.files)}): {z.files[:8]}")
    for k in z.files[:4]:
        v = z[k]
        if hasattr(v, "shape"):
            print(f"    {k}: shape={v.shape} dtype={v.dtype}")
        else:
            print(f"    {k}: {type(v).__name__} = {str(v)[:100]}")
    # 找字表
    for k in z.files:
        v = z[k]
        if hasattr(v, "shape") and v.dtype.kind in ("U", "S", "O"):
            print(f"  [候选字表] {k}: {v.shape} 前10={list(v[:10])}")
            print(f"     总数 {len(v)}, '阜' 在里面吗: {'阜' in [str(x) for x in v]}")
            break

print()
print("=== 训练/标准字形库里有没有 '阜' ===")
import csv

for csvp in ("assets/train_50k_v2.csv", "assets/eval_v13_strict.csv"):
    if not os.path.exists(csvp):
        continue
    rows = list(csv.DictReader(open(csvp, encoding="utf-8")))
    chars = set(r["character"] for r in rows)
    print(f"  {csvp}: {len(rows)} 行, {len(chars)} 个字, '阜'在内={'阜' in chars}")
    k_rows = [r for r in rows if r["character"] == "阜"]
    print(f"      '阜' 的样本数={len(k_rows)}")
    if k_rows:
        print(f"      script={set(r['script'] for r in k_rows)}")
        print(f"      std_path 样例={k_rows[0].get('std_path')}")
