"""诊断老数据里"一个字对多个 glyph_id"的规律。"""
import csv, collections, os
os.chdir("/root/Workspace/xy/DiT")

o = list(csv.DictReader(open("assets/train_fame-kxl-tj-px60.csv", encoding="utf-8")))
w = list(csv.DictReader(open("assets/train_hcsu_kxl.csv", encoding="utf-8")))

m = collections.defaultdict(set)
for r in o:
    m[r["character"]].add(r["glyph_id"])
multi = {k: v for k, v in m.items() if len(v) > 1}
print(f"老数据: 字 {len(m)}, 一字多 glyph_id 的 {len(multi)}")

# 1) 多 id 是否与书体一一对应
by_script = 0
for c in multi:
    rows = [r for r in o if r["character"] == c]
    g2s = collections.defaultdict(set)
    for r in rows:
        g2s[r["glyph_id"]].add(r["script"])
    ok = all(len(v) == 1 for v in g2s.values())
    ok = ok and len({next(iter(v)) for v in g2s.values()}) == len(g2s)
    by_script += ok
print(f"  其中「多 id 与书体一一对应」的: {by_script} ({100*by_script/len(multi):.1f}%)")

# 2) 多 id 是否与 (书体,书家) 一一对应
by_cs = 0
for c in multi:
    rows = [r for r in o if r["character"] == c]
    g2s = collections.defaultdict(set)
    for r in rows:
        g2s[r["glyph_id"]].add((r["script"], r["calligrapher"]))
    ok = all(len(v) == 1 for v in g2s.values())
    by_cs += ok
print(f"  其中「多 id 与 (书体,书家) 一一对应」的: {by_cs} ({100*by_cs/len(multi):.1f}%)")

# 3) 具体例子
for c in list(multi)[:3]:
    rows = [r for r in o if r["character"] == c]
    print(f"  例 [{c}] 共 {len(rows)} 行:")
    seen = set()
    for r in rows:
        key = (r["glyph_id"], r["character_id"])
        if key in seen:
            continue
        seen.add(key)
        n = sum(1 for x in rows if (x["glyph_id"], x["character_id"]) == key)
        sc = sorted({x["script"] for x in rows if (x["glyph_id"], x["character_id"]) == key})
        cg = sorted({x["calligrapher"] for x in rows if (x["glyph_id"], x["character_id"]) == key})
        print(f"     glyph_id={r['glyph_id']:>6} character_id={r['character_id']:>6} "
              f"n={n:>3} 书体={sc} 书家={cg[:3]}{'...' if len(cg)>3 else ''}")

# 4) 关键: 模型只用 glyph_id 当 y_char (v12 是 no_char_cond, 不用)。
#    但 num_characters 会被这个虚高。
print()
print(f"glyph_id 总数(老) {len({r['glyph_id'] for r in o})}  vs 字数 {len(m)}")
print(f"glyph_id 总数(合并) {len({r['glyph_id'] for r in o+w})}  vs 字数 "
      f"{len({r['character'] for r in o+w})}")

# 5) 合并后是否仍有冲突
mm = collections.defaultdict(set)
for r in o + w:
    mm[r["character"]].add(r["glyph_id"])
print(f"合并后: 一字多 glyph_id 的 {sum(1 for v in mm.values() if len(v) > 1)} / {len(mm)}")
