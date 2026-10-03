import csv

rows = list(csv.DictReader(open("assets/train_50k_v2_fixed.csv",
                                encoding="utf-8")))
for col in ("calligrapher_id", "script_id", "character_id", "glyph_id"):
    v = [int(r[col]) for r in rows if r.get(col, "").strip()]
    print(f"  {col}: 范围 [{min(v)}, {max(v)}]  唯一 {len(set(v))}")

# 字表的 idx（dataset 里 char_vocab 是 0..n-1）
chars = sorted(set(r["character"] for r in rows))
print(f"  char_vocab 大小: {len(chars)} -> idx 范围 [0, {len(chars)-1}]")

need = max(max(int(r["calligrapher_id"]) for r in rows),
           max(int(r["script_id"]) for r in rows),
           len(chars) - 1) + 1
print(f"\n  num_classes 至少需要: {need}")
