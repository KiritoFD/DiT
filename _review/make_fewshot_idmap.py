"""为 few-shot 生成扩展的 callig_id_map，并让 CSV / 配置一致。

callig_id_map 的真实格式（实测）:
    {"num_calligraphers": 45, "id_map": {"<原始书家id>": <紧凑id 0..44>, ...}}
id_map 是 **原始 id -> 紧凑 id** 的映射。eval-cache 的护栏检查的就是这个。

所以新书家需要:
    1) CSV 里给一个**原始 id**（用 9999）
    2) id_map 加 "9999" -> 45
    3) num_calligraphers -> 46
"""
import csv
import json
import os

os.chdir("/root/Workspace/xy/DiT")
SRC = "assets/callig_id_map_50k.json"
base = json.load(open(SRC, encoding="utf-8"))
print(f"  原表: num_calligraphers={base.get('num_calligraphers')}, "
      f"id_map {len(base.get('id_map', {}))} 条")

CALS = ("怀素", "伊秉绶", "徐渭", "沈周")
RAW_ID, NEW_ID = 9999, 45

for c in CALS:
    mm = {"num_calligraphers": 46, "id_map": dict(base["id_map"])}
    mm["id_map"][str(RAW_ID)] = NEW_ID
    out = f"assets/callig_id_map_50k_fs_{c}.json"
    json.dump(mm, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # CSV 的原始 id 统一改成 9999
    for name in ("train", "eval"):
        p = f"assets/fs_{c}_{name}.csv"
        rows = list(csv.DictReader(open(p, encoding="utf-8")))
        cols = list(rows[0].keys())
        for r in rows:
            r["calligrapher_id"] = str(RAW_ID)
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)

    cp = f"src/train/configs/v13_fs_{c}.json"
    d = json.load(open(cp, encoding="utf-8"))
    d["callig_id_map"] = out
    d["num_calligraphers"] = 46
    json.dump(d, open(cp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"  {c}: {out}  (id_map 加 {RAW_ID}->{NEW_ID}, n_cal=46); "
          f"CSV calligrapher_id={RAW_ID}; 配置已指向")
