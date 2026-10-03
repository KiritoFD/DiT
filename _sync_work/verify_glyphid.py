"""关键：模型用的是 glyph_id（决定 g 条件）。查 glyph_id 是否也有一对多。"""
import csv
import os
from collections import Counter, defaultdict

os.chdir("/root/Workspace/xy/DiT")
rows = list(csv.DictReader(open("assets/train_50k_v2.csv", encoding="utf-8")))

g2c = defaultdict(set)
c2g = defaultdict(set)
for r in rows:
    g2c[int(r["glyph_id"])].add(r["character"])
    c2g[r["character"]].add(int(r["glyph_id"]))

print("  === glyph_id 的一致性 ===")
print(f"  唯一 glyph_id : {len(g2c)}")
print(f"  唯一 character: {len(c2g)}")
bad1 = {k: v for k, v in g2c.items() if len(v) > 1}
bad2 = {k: v for k, v in c2g.items() if len(v) > 1}
print(f"  ⚠ 一个 glyph_id 对应多个字符: {len(bad1)}")
for k in list(bad1)[:5]:
    print(f"      glyph_id {k} -> {sorted(bad1[k])}")
print(f"  ⚠ 一个字符对应多个 glyph_id: {len(bad2)}")
for k in list(bad2)[:5]:
    print(f"      {k!r} -> glyph_ids {sorted(bad2[k])}")

print()
print("  === 是否等于 character_id ===")
same = sum(1 for r in rows if r["glyph_id"] == r["character_id"])
print(f"    glyph_id == character_id 的行: {same} / {len(rows)}")

print()
print("  === glyph_id 口径下的 (书家,字) 对 ===")
pair = Counter((int(r["calligrapher_id"]), int(r["glyph_id"])) for r in rows)
d1 = sum(1 for v in pair.values() if v == 1)
print(f"    对数 {len(pair)}, 只 1 张 {d1} ({d1/len(pair)*100:.1f}%)")
nc = len(g2c)
ncal = len(set(int(r["calligrapher_id"]) for r in rows))
print(f"    {ncal} 书家 x {nc} 字 = {ncal*nc} 组合, "
      f"覆盖 {len(pair)} ({len(pair)/(ncal*nc)*100:.2f}%), "
      f"未监督 {(1-len(pair)/(ncal*nc))*100:.1f}%")

print()
print("  === 对 strict 集的影响（strict 是未见的三元组）===")
st = list(csv.DictReader(open("assets/eval_v13_strict.csv", encoding="utf-8")))
# 该字的 g（标准字形）在训练集里出现过吗
tr_glyphs = set(int(r["glyph_id"]) for r in rows)
miss = sum(1 for r in st if int(r["glyph_id"]) not in tr_glyphs)
print(f"    strict {len(st)} 条里，glyph_id 在训练集**没见过**的: {miss}")
# 该 glyph_id 被多少书家写过
gc = defaultdict(set)
for r in rows:
    gc[int(r["glyph_id"])].add(int(r["calligrapher_id"]))
cnt = [len(gc.get(int(r["glyph_id"]), set())) for r in st]
import statistics as S
print(f"    strict 的 g 被多少书家写过: min={min(cnt)} med={int(S.median(cnt))} max={max(cnt)}")
