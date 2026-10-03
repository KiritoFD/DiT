"""50k 的 45 个书家是否是 callig_id_map_base 的子集？决定预训练书家表能否直接取子集。"""
import csv
import json
import os

import torch

os.chdir("/root/Workspace/xy/DiT")

m = json.load(open("assets/callig_id_map_base.json", encoding="utf-8"))
base_map = {int(k): int(v) for k, v in m["id_map"].items()}
print(f"  base: num_calligraphers={m['num_calligraphers']}  条数={len(base_map)}")
print(f"  base raw ids: {sorted(base_map)[:10]} ...")

rows = list(csv.DictReader(open("assets/train_50k.csv", encoding="utf-8")))
new_ids = sorted(set(int(r["calligrapher_id"]) for r in rows))
print(f"  50k : {len(new_ids)} 个书家, raw ids {new_ids[:10]} ...")

missing = [i for i in new_ids if i not in base_map]
print()
if missing:
    print(f"  ✗ 不在 base 里: {len(missing)} 个 -> {missing[:10]}")
else:
    print("  ✓ 全部在 base 里 —— 可直接按新顺序取预训练表的子集")

# 名字核对
name_by_raw = {}
for r in rows:
    name_by_raw.setdefault(int(r["calligrapher_id"]), r["calligrapher"])
t = torch.load("assets/callig_emb_pretrained_base.pt", map_location="cpu")
print(f"  预训练表: {tuple(t.shape) if hasattr(t, 'shape') else type(t)}")
