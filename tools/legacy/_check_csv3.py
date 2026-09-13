import os, sys, csv
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
rows = list(csv.DictReader(open(r"G:\GitHub\DiT\5script\train_top30_clean.csv", encoding="utf-8")))
from collections import Counter
sc = Counter(r["script"] for r in rows)
print("CSV script 分布:")
for s, c in sc.most_common():
    print(f"  {s}: {c}")
print()
print("前5行:")
for r in rows[:5]:
    print(f"  char={r['character']} script={r['script']} path={r['image_path']}")
# 看图片是否就是 MCCD 里的某张
# CSV path: final_imgs_256/xxx.png — 远程的, 本地没有
# 但 MCCD 文件名: 字-书体-朝代-来源-编号.png
# 需要建立映射
