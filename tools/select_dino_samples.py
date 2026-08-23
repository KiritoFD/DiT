"""选取适合做 DINO embedding 验证的样本：
   挑选同一字符、多位书家写过的样本，用于验证 DINO 能否区分字符内容 vs 书家风格。"""
import csv, os
from collections import defaultdict

CSV = "5script/train_top30_clean.csv"
rows = []
with open(CSV, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        rows.append(r)

# 按 (script_id, character) 分组，找 calligrapher 最多的字
char_calligs = defaultdict(lambda: defaultdict(list))  # (sid, char) -> {callig_id: [image_path,...]}
for r in rows:
    key = (int(r["script_id"]), r["character"])
    cid = r["calligrapher"]
    char_calligs[key][cid].append(r["image_path"])

# 排序：calligrapher 数最多的字
ranked = []
for (sid, char), calligs in char_calligs.items():
    if len(calligs) >= 5:  # 至少5位书家写过
        ranked.append((len(calligs), sid, char, calligs))
ranked.sort(key=lambda x: -x[0])

# 选 top 20 个字，每个取每位书家的第1张图
selected = []
for n_calligs, sid, char, calligs in ranked[:20]:
    for cname, paths in sorted(calligs.items()):
        selected.append({
            "image_path": paths[0],
            "character": char,
            "calligrapher": cname,
            "script_id": sid,
            "char_id": None,  # filled below
        })

# 补充 char_id
char2id = {}
for r in rows:
    char2id[r["character"]] = r["character_id"]
for s in selected:
    s["char_id"] = char2id.get(s["character"])

print(f"选取 {len(selected)} 张图片，覆盖 {len(set(s['character'] for s in selected))} 个字符")
print(f"\nTop 字符（按书家数排序）:")
for n, sid, char, calligs in ranked[:20]:
    script = {0:"楷",1:"篆",2:"草",3:"行",4:"隶"}[sid]
    print(f"  {script} {char} (U+{ord(char):05X}): {n} 位书家 — {', '.join(sorted(calligs.keys())[:6])}...")

# 写成 csv 供拉取
out = "tools/_dino_sample.csv"
with open(out, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["image_path","character","calligrapher","script_id","char_id"])
    w.writeheader()
    w.writerows(selected)
print(f"\n样本清单 → {out}")
