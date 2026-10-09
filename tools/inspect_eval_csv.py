import os
import csv
import json

base_dir = "/root/Workspace/xy/DiT"
eval_csv = os.path.join(base_dir, "exp-std/csv/eval200_fixed.csv")

print(f"=== Inspecting {eval_csv} ===")
rows = []
with open(eval_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

print(f"Total rows: {len(rows)}")
first = rows[0]
print(f"Columns: {list(first.keys())}")

calligs = sorted(list(set(r["calligrapher"] for r in rows)))
print(f"Unique calligraphers ({len(calligs)}): {calligs}")

scripts = sorted(list(set(r["script"] for r in rows)))
print(f"Unique scripts ({len(scripts)}): {scripts}")

# check if std_path files exist
missing_std = 0
for r in rows:
    sp = os.path.join(base_dir, r["std_path"])
    if not os.path.exists(sp):
        missing_std += 1
print(f"Missing std_path: {missing_std} / {len(rows)}")

# check calligrapher_id, script_id, character_id min/max
cid_list = [int(r["calligrapher_id"]) for r in rows if r["calligrapher_id"]]
chid_list = [int(r["character_id"]) for r in rows if r["character_id"]]
print(f"calligrapher_id range: min={min(cid_list)}, max={max(cid_list)}")
print(f"character_id range: min={min(chid_list)}, max={max(chid_list)}")
print(f"Unique char_ids: {len(set(chid_list))}")

# Check v66 / v68 resolved_config to see how eval was run for them
for exp in ["v66_tables_condroute2456", "v68_aug_sp_c2ot", "v70_aug_sp_stdskel_c2ot"]:
    cfg_path = os.path.join(base_dir, f"src/train/configs/{exp}.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            c = json.load(f)
        print(f"[{exp}] callig_map: {c.get('callig_id_map')}, char_map: {c.get('char_id_map')}, num_callig: {c.get('num_calligraphers')}, num_char: {c.get('num_characters')}")
