import os
import glob
import json

candidates = [
    "assets/callig_id_map.json",
    "assets/callig_id_map_41.json",
    "assets/callig_script_id_map.json",
    "assets/char_id_map.json",
    "assets/char_to_id.json",
    "assets/callig_to_id.json",
    "exp-std/assets/callig_script_id_map.json",
]

for cand in candidates:
    full = os.path.join("/root/Workspace/xy/DiT", cand)
    if os.path.exists(full):
        try:
            with open(full, "r", encoding="utf-8") as f:
                d = json.load(f)
            print(f"[FOUND] {cand} -> type={type(d)}, len={len(d)}")
            if isinstance(d, dict):
                items = list(d.items())[:3]
                print(f"        samples: {items}")
            elif isinstance(d, list):
                print(f"        samples: {d[:3]}")
        except Exception as e:
            print(f"[ERROR] {cand}: {e}")
    else:
        print(f"[MISSING] {cand}")
