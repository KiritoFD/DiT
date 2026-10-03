import csv
import json
import os

fs = sorted(os.listdir("data/50k/std"))[:8]
print("  样例文件:", fs)
print(f"  总数: {len(os.listdir('data/50k/std'))}")
print()

# csv
rows = list(csv.DictReader(open("assets/train_50k_v2.csv", encoding="utf-8")))
sp_chars = set()
for r in rows[:200]:
    sp_chars.add((r["script"], r["character"]))
print(f"  csv 前200 (script,char) 样例: {list(sp_chars)[:8]}")

# glyph_id 与 char 的对应（建 char->glyph_id 映射）
m = {}
for r in rows:
    m[(r["script"], r["character"])] = (r["glyph_id"], r["character_id"])
print(f"  (script,char) -> (glyph_id, char_id) 样例: {list(m.items())[:3]}")

# v13 config
d = json.load(open("src/train/configs/v13_base_50k.json", encoding="utf-8"))
print(f"  chars_per_script={d.get('chars_per_script')}")
print(f"  glyph_inject_mode={d.get('glyph_inject_mode')}")
