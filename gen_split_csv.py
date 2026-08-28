# -*- coding: utf-8 -*-
"""生成切分 csv：train/test/eval，引用 final_images/{img_id}.png + 三元组 id。
输出 final_train.csv / final_test.csv / final_eval.csv
"""
import json, sys, csv
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

m = json.load(open("final_manifest_split.json", encoding="utf-8"))
print("manifest:", len(m))

header = ["image_path", "calligrapher", "script", "character",
          "calligrapher_id", "script_id", "character_id"]

splits = {"train": [], "test": [], "eval": []}
for r in m:
    fs = r["final_split"]
    splits[fs].append({
        "image_path": f"final_images/{r['img_id']}.png",
        "calligrapher": r["orig_calli"],
        "script": r["orig_script"],
        "character": r["orig_char"],
        "calligrapher_id": r["calli_id"],
        "script_id": r["script_id"],
        "character_id": r["char_id"],
    })

for name, rows in splits.items():
    out = f"final_{name}.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    print(f"{out}: {len(rows)} rows")

print("total:", sum(len(v) for v in splits.values()))
