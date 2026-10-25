import csv
from pathlib import Path

r5 = list(csv.DictReader(open("exp-std/csv/eval_top5_fixed.csv", encoding="utf-8-sig")))
r10 = list(csv.DictReader(open("exp-std/csv/eval200_fixed.csv", encoding="utf-8-sig")))
r30 = list(csv.DictReader(open("exp-std/csv/eval_top5_30.csv", encoding="utf-8-sig")))

img_to_r10 = {r["image_path"]: (i, r) for i, r in enumerate(r10)}

mapping = []
for i30, r in enumerate(r30):
    p = r["image_path"]
    orig_i5 = int(r["orig_eval_idx"])
    i10 = img_to_r10[p][0] if p in img_to_r10 else -1
    mapping.append({
        "i30": i30,
        "orig_i5": orig_i5,
        "i10": i10,
        "char": r["character"],
        "callig": r["calligrapher"],
        "script": r["script"],
        "is_seen": r["is_seen"]
    })

print(f"{'#':2s} | {'i5':3s} | {'i10':3s} | {'Char':4s} | {'Callig':6s} | {'Scr':2s} | {'Status':6s}")
print("-" * 45)
for m in mapping:
    st = m["is_seen"].upper()
    print(f"{m['i30']:02d} | {m['orig_i5']:02d} | {m['i10']:03d} | {m['char']:2s}   | {m['callig']:6s} | {m['script']:2s}  | {st:6s}")
