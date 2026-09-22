"""查「陞/升/昇」在训练集里的出现次数，以及 strict/seen 里的情况。"""
import csv
import os
from collections import Counter

os.chdir("/root/Workspace/xy/DiT")

TARGETS = ["升", "陞", "昇", "陹", "阩"]
FILES = [("train", "assets/train_50k_v2_fixed.csv"),
         ("train_orig", "assets/train_50k_v2.csv"),
         ("seen", "assets/eval_v13_seen_fixed.csv"),
         ("strict", "assets/eval_v13_strict_fixed.csv")]

for tag, f in FILES:
    if not os.path.exists(f):
        print(f"  {tag}: {f} 不存在")
        continue
    rows = list(csv.DictReader(open(f, encoding="utf-8")))
    cc = Counter(r["character"] for r in rows)
    print(f"\n  === {tag} ({f}) {len(rows)} 条 ===")
    for t in TARGETS:
        n = cc.get(t, 0)
        if n:
            ex = [r for r in rows if r["character"] == t][:3]
            det = "; ".join(f"{r['calligrapher']}/{r['script']} "
                            f"src={os.path.basename(r.get('src_image_path',''))}"
                            for r in ex)
            print(f"    {t}: {n} 条   {det}")
        else:
            print(f"    {t}: 0 条")

# 训练集里所有"含 升/陞/昇 部件"的字
print(f"\n  === 训练集里含这些部件的字 ===")
rows = list(csv.DictReader(open("assets/train_50k_v2_fixed.csv",
                                encoding="utf-8")))
cc = Counter(r["character"] for r in rows)
hits = {c: n for c, n in cc.items()
        if any(t in c for t in ("升", "陞", "昇"))}
for c, n in sorted(hits.items(), key=lambda x: -x[1]):
    print(f"    {c}: {n}")

print(f"\n  === 训练集的字符总数 / 唯一数 ===")
print(f"    总条数: {len(rows)}")
print(f"    唯一字符: {len(cc)}")
rare = [(c, n) for c, n in cc.items() if n <= 2]
print(f"    出现 <=2 次的字符: {len(rare)} 个")
print(f"    出现 ==1 次的字符: {sum(1 for c,n in cc.items() if n==1)} 个")
