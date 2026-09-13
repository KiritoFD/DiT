import csv, sys, os, json, random
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = "/root/Workspace/xy/DiT"

# 1) 当前 eval100_top30_clean.csv: 79 行 (clean 过的)
clean = list(csv.DictReader(open(os.path.join(BASE, "5script", "eval100_top30_clean.csv"), encoding="utf-8")))
# 2) 原始 eval100_top30.csv: 100 行
orig100 = list(csv.DictReader(open(os.path.join(BASE, "5script", "eval100_top30.csv"), encoding="utf-8")))
# 3) eval.csv: 472 行
eval472 = list(csv.DictReader(open(os.path.join(BASE, "5script", "eval.csv"), encoding="utf-8")))

print(f"clean eval: {len(clean)}")
print(f"orig100: {len(orig100)}")
print(f"eval.csv: {len(eval472)}")

# clean eval 的 image_path 是什么前缀?
print("\nclean paths sample:")
for r in clean[:3]:
    print(f"  {r['image_path']}")
print("\neval.csv paths sample:")
for r in eval472[:3]:
    print(f"  {r['image_path']}")

# 看 eval.csv 是否有 glyph_id
print("\neval.csv columns:", list(eval472[0].keys()))
# 看 eval.csv 的 script 分布
from collections import Counter
sc = Counter(r["script"] for r in eval472)
print("eval.csv scripts:", dict(sc))

# 看 clean 79 行是从 100 行里删了哪些 (21 行)
clean_paths = {r["image_path"] for r in clean}
orig100_paths = {r["image_path"] for r in orig100}
removed = orig100_paths - clean_paths
print(f"\nremoved {len(removed)} from orig100:")
# 找它们的 script
for r in orig100:
    if r["image_path"] in removed:
        print(f"  {r['image_path']} script={r['script']} char={r['character']}")

# 看 eval472 中有多少 image_path 在 final_imgs_256 存在
import os
missing = 0
existing = 0
for r in eval472:
    p = r["image_path"]
    if p.startswith("final_images/"):
        p = p.replace("final_images/", "final_imgs_256/", 1)
    full = os.path.join(BASE, p)
    if os.path.isfile(full):
        existing += 1
    else:
        missing += 1
print(f"\neval.csv: {existing} exist, {missing} missing")
