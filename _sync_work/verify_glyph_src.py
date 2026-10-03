"""查同一个字的多个 glyph_id 是不是指向不同的标准字形文件。"""
import csv
import hashlib
import os
from collections import defaultdict

os.chdir("/root/Workspace/xy/DiT")
rows = list(csv.DictReader(open("assets/train_50k_v2.csv", encoding="utf-8")))

byc = defaultdict(set)
for r in rows:
    byc[r["character"]].add((int(r["glyph_id"]), r["std_path"]))

print("  === 例：同字多 glyph_id 的 std_path ===")
for ch in ("㑺", "䟽", "㝡", "㱕"):
    if ch not in byc:
        continue
    print(f"  {ch!r}:")
    for gid, p in sorted(byc[ch]):
        full = os.path.join("/root/Workspace/xy/DiT", p)
        if os.path.exists(full):
            h = hashlib.md5(open(full, "rb").read()).hexdigest()[:12]
            print(f"     glyph_id={gid:<6} {p}  md5={h}  {os.path.getsize(full)}B")
        else:
            print(f"     glyph_id={gid:<6} {p}  ✗ 文件不存在")

print()
print("  === 全局：同字多 glyph_id 时，std_path 也不同吗 ===")
multi = {k: v for k, v in byc.items() if len(v) > 1}
same_path = 0
diff_path = 0
for ch, s in multi.items():
    paths = set(p for _, p in s)
    if len(paths) == 1:
        same_path += 1
    else:
        diff_path += 1
print(f"    同字多 id: {len(multi)}")
print(f"      但 std_path 相同: {same_path}   ← 同一个字形，只是 id 重复（无害）")
print(f"      std_path 不同: {diff_path}     ← 真的是不同字形（有害）")

print()
print("  === std_path 的命名规律（判断是否含字体/来源）===")
import random
samples = random.sample(rows, 8)
for r in samples:
    print(f"    char={r['character']} gid={r['glyph_id']:<6} {r['std_path']}")
