import csv, json, collections, os
os.chdir("/root/Workspace/xy/DiT")

rows = list(csv.DictReader(open("assets/train_fame-kxl-tj-px60.csv", encoding="utf-8")))
print("行数", len(rows))
print("script -> script_id:")
print(" ", dict(collections.Counter((r["script"], r["script_id"]) for r in rows)))

# calligrapher -> id
c2i = {}
for r in rows:
    c2i.setdefault(r["calligrapher"], set()).add(r["calligrapher_id"])
print(f"\n书家 {len(c2i)} 个, id 范围 "
      f"{min(int(r['calligrapher_id']) for r in rows)}~{max(int(r['calligrapher_id']) for r in rows)}")
print(" 多 id 的书家:", {k: v for k, v in c2i.items() if len(v) > 1})

# glyph_id / character_id 关系
g2c = collections.defaultdict(set)
for r in rows[:5000]:
    g2c[r["glyph_id"]].add(r["character"])
print(f"\nglyph_id 是否唯一对应 character: {all(len(v)==1 for v in g2c.values())}")
print(" 样例 (glyph_id -> char):", list(g2c.items())[:4])
cid = {r["character"]: r["character_id"] for r in rows}
gid = {r["character"]: r["glyph_id"] for r in rows}
print(f" character_id 范围 {min(map(int,cid.values()))}~{max(map(int,cid.values()))}  个数 {len(set(cid.values()))}")
print(f" glyph_id     范围 {min(map(int,gid.values()))}~{max(map(int,gid.values()))}  个数 {len(set(gid.values()))}")
print(f" 同一字有多个 character_id? {len(cid) != len(set(cid.values()))}")
print(f" 同一字有多个 glyph_id?     {len(gid) != len(set(gid.values()))}")

# 是否有 glyph 词表文件
print("\n候选词表文件:")
for d in ["assets", "data", "configs", "labels"]:
    if os.path.isdir(d):
        for f in os.listdir(d):
            if any(k in f.lower() for k in ["glyph", "char", "vocab", "uni"]):
                print(f"  {d}/{f}")
