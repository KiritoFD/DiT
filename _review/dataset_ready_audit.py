"""新数据集可用性审计: 字表 / std 骨架覆盖 / CSV / 图像文件。

合并数据 = 老 (train_fame-kxl-tj-px60.csv) + HCSU (train_hcsu_kxl.csv)
两套的 std 骨架在不同目录, 训练时需要把两边都配进 std_path (CSV 里已写各自路径)。
"""
import csv, os, collections, json
import numpy as np
from PIL import Image

os.chdir("/root/Workspace/xy/DiT")

OLD = "assets/train_fame-kxl-tj-px60.csv"
NEW = "assets/train_hcsu_kxl.csv"


def load(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


o, w = load(OLD), load(NEW)
m = o + w

print("=" * 78)
print("1. 规模")
print(f"   老数据 {len(o):>6} 张   字 {len({r['character'] for r in o}):>5}   "
      f"(书体,字) {len({(r['script'], r['character']) for r in o}):>5}")
print(f"   HCSU   {len(w):>6} 张   字 {len({r['character'] for r in w}):>5}   "
      f"(书体,字) {len({(r['script'], r['character']) for r in w}):>5}")
print(f"   合并   {len(m):>6} 张   字 {len({r['character'] for r in m}):>5}   "
      f"(书体,字) {len({(r['script'], r['character']) for r in m}):>5}")

och = {r["character"] for r in o}
wch = {r["character"] for r in w}
print(f"   字: 老 {len(och)} + HCSU {len(wch)} -> 合并 {len(och | wch)}  "
      f"(新增 {len(wch - och)}, 重叠 {len(och & wch)})")

print("=" * 78)
print("2. 文件存在性")
for tag, rows in [("老", o), ("HCSU", w), ("合并", m)]:
    mi = sum(1 for r in rows if not os.path.exists(r["image_path"]))
    ms = sum(1 for r in rows if not os.path.exists(r["std_path"]))
    print(f"   {tag:4s} {len(rows):>6} 行:  缺 image_path {mi}   缺 std_path {ms}")

print("=" * 78)
print("3. std 骨架覆盖 (按 (书体,字) 组合看)")
for tag, rows in [("老", o), ("HCSU", w)]:
    need = {(r["script"], r["character"]) for r in rows}
    have = set()
    for r in rows:
        if os.path.exists(r["std_path"]):
            have.add((r["script"], r["character"]))
    miss = need - have
    print(f"   {tag:4s} 需要 {len(need):>5} 个 (书体,字), 有骨架 {len(have):>5}, "
          f"缺失 {len(miss)}")
    if miss:
        print(f"        缺样例: {sorted(miss)[:8]}")

# 合并后: 老与 HCSU 的 (书体,字) 覆盖互补情况
oneed = {(r["script"], r["character"]) for r in o}
wneed = {(r["script"], r["character"]) for r in w}
print(f"   合并需要 {len(oneed | wneed)} 个 (书体,字) 组合")
print(f"     仅老数据有: {len(oneed - wneed)}")
print(f"     仅 HCSU 有: {len(wneed - oneed)}")
print(f"     两边都有:   {len(oneed & wneed)}")

print("=" * 78)
print("4. std 骨架质量抽查 (256x256 / 二值 / 墨点率)")
import random
random.seed(0)
for tag, rows in [("老", o), ("HCSU", w)]:
    samp = random.sample(rows, 300)
    bad_sz = bad_bin = 0
    inks = []
    for r in samp:
        try:
            a = np.asarray(Image.open(r["std_path"]).convert("L"))
        except Exception:
            bad_sz += 1
            continue
        if a.shape != (256, 256):
            bad_sz += 1
        if len(np.unique(a)) > 2:
            bad_bin += 1
        inks.append(float((a < 128).mean()))
    print(f"   {tag:4s} 抽 300:  尺寸非256 {bad_sz}   非二值 {bad_bin}   "
          f"墨点率 med={np.median(inks):.4f}")

print("=" * 78)
print("5. 训练可直接用的关键前提")
# 书家 id 是否都在词表里
cm = json.load(open("assets/callig_id_map_hcsu.json", encoding="utf-8"))
mp = cm["id_map"]
bad = {r["calligrapher"] for r in m if r["calligrapher_id"] not in mp}
print(f"   callig 词表 num_calligraphers={cm['num_calligraphers']}, "
      f"未覆盖书家 {sorted(bad) if bad else '无'}")
# glyph_id 唯一性
g2c = collections.defaultdict(set)
for r in m:
    g2c[r["glyph_id"]].add(r["character"])
conflict = {g: v for g, v in g2c.items() if len(v) > 1}
print(f"   glyph_id 数 {len(g2c)}, 一个 id 对多字的冲突 {len(conflict)}")
c2g = collections.defaultdict(set)
for r in m:
    c2g[r["character"]].add(r["glyph_id"])
print(f"   一个字对多 glyph_id 的: {sum(1 for v in c2g.values() if len(v) > 1)}")
print(f"   书体 script_id: {sorted({(r['script'], r['script_id']) for r in m})}")
# 每书家字数
cch = collections.defaultdict(set)
for r in m:
    cch[r["calligrapher"]].add(r["character"])
per = sorted(len(v) for v in cch.values())
print(f"   每书家字数: min={per[0]} med={per[len(per)//2]} max={per[-1]}")
