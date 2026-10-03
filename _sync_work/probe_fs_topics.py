"""few-shot 选题体检：书家是否真的未训过 + train/eval 是否按字互斥 + std_path 是否可用。

为什么要写这个: 交接文档 74 的 B 组候选(王羲之/欧阳询/颜真卿...)是"未用过的**文件夹**",
但同一书家可能有另一个白底文件夹**已经**进了 50k 训练表 —— 那种书家做 few-shot
不是"全新书家",结论会被污染。这里按书家名逐一对账。
"""
import collections
import csv
import json
import os

os.chdir("/root/Workspace/xy/DiT")

CANDS = ["沈周", "伊秉绶", "傅山", "徐渭", "宋高宗", "怀素", "王羲之", "钟繇", "欧阳询",
         "虞世南", "董其昌", "柳公权", "王宠", "朱熹", "王献之", "鲜于枢", "黄庭坚",
         "颜真卿", "赵孟頫", "祝允明", "蔡襄", "米芾", "苏轼", "文征明"]

m = json.load(open("assets/callig_script_id_map.json", encoding="utf-8"))
print(f"[表] num_calligraphers={m['num_calligraphers']} num_pairs={m['num_pairs']}")

rows = list(csv.DictReader(open("assets/train_50k_v2.csv", encoding="utf-8")))
id2name = {}
for r in rows:
    id2name.setdefault(r["calligrapher_id"], r["calligrapher"])
name2scripts = collections.defaultdict(set)
for key in m["pair_map"]:
    cid, sid = key.split(":")
    name2scripts[id2name.get(cid, "?" + cid)].add(sid)
print(f"[表] 已训书家 {len(name2scripts)} 人")

print("\n[候选] 是否已在训练表里 (IN = 不能当全新书家用)")
for c in CANDS:
    tag = "IN   " + str(sorted(name2scripts[c])) if c in name2scripts else "NEW"
    print(f"  {c:5s} {tag}")

print("\n[已训书家名单] " + " ".join(sorted(name2scripts)))

# 50k 里每个字的 std 骨架图（few-shot 借 std 用）
char2std = collections.defaultdict(list)
for r in rows:
    if r.get("std_path"):
        char2std[r["character"]].append(r["std_path"])
print(f"\n[std] 50k 里有 {len(char2std)} 个不同的字有 std 骨架图")

for cal in ["沈周", "伊秉绶", "徐渭", "宋高宗", "怀素"]:
    tp, ep = f"assets/fs50_{cal}_train.csv", f"assets/fs50_{cal}_eval.csv"
    if not os.path.exists(tp):
        continue
    tr = list(csv.DictReader(open(tp, encoding="utf-8")))
    ev = list(csv.DictReader(open(ep, encoding="utf-8")))
    tc = {r["character"] for r in tr}
    ec = {r["character"] for r in ev}
    overlap = tc & ec
    ids = [r["img_id"] for r in tr + ev]
    missing_std = [r["character"] for r in tr + ev if not os.path.isfile(r["std_path"])]
    print(f"\n[{cal}] train={len(tr)} eval={len(ev)} "
          f"train字={len(tc)} eval字={len(ec)} 字重叠={len(overlap)} "
          f"img_id唯一={len(set(ids)) == len(ids)} img_id范围="
          f"{min(int(i) for i in ids)}..{max(int(i) for i in ids)} "
          f"std缺失={len(missing_std)}")
    if overlap:
        print(f"    ⚠ 重叠字(前20): {''.join(sorted(overlap)[:20])}")
