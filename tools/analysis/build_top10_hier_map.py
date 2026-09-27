#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import json

df = pd.read_csv("assets/train_top10_style23.csv")
cs = df.groupby(["calligrapher", "calligrapher_id", "script", "script_id", "pair_id", "slot_name"]).size().reset_index(name="count")
print(cs.to_string())

# 按照 pretrain_callig_script_emb 的标准构建映射字典
top10_cals = sorted(df["calligrapher"].unique())
cal2idx = {c: i for i, c in enumerate(top10_cals)}

callig_map = {}
for c in top10_cals:
    cid = int(df[df["calligrapher"] == c]["calligrapher_id"].iloc[0])
    callig_map[cid] = cal2idx[c]

pair_map = {}
pair_to_callig = [0] * len(cs)
pair_counts = {}

for _, r in cs.iterrows():
    pid = int(r["pair_id"])
    cid = int(r["calligrapher_id"])
    sid = int(r["script_id"])
    k = f"{cid}:{sid}"
    pair_map[k] = pid
    pair_to_callig[pid] = cal2idx[r["calligrapher"]]
    pair_counts[str(pid)] = int(r["count"])

meta = {
    "num_pairs": len(cs),
    "num_calligraphers": len(top10_cals),
    "pair_map": pair_map,
    "callig_map": {str(k): v for k, v in callig_map.items()},
    "pair_to_callig": pair_to_callig,
    "pair_counts": pair_counts,
    "min_samples": 30
}

out_p = "assets/callig_script_id_map_top10_hier.json"
with open(out_p, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"\n已生成层级 SupCon 专用的映射配置: {out_p}")
