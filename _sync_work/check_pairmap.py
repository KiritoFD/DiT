import json
import os

os.chdir("/root/Workspace/xy/DiT")
m = json.load(open("assets/callig_script_id_map.json", encoding="utf-8"))
print("  keys:", list(m))
print("  num_calligraphers =", m.get("num_calligraphers"))
print("  num_pairs         =", m.get("num_pairs"))
print("  min_samples       =", m.get("min_samples"))

pm = m.get("pair_map") or {}
print("\n  pair_map 条数 =", len(pm))
for k, v in list(pm.items())[:6]:
    print(f"    {k!r} -> {v!r}")

cm = m.get("callig_map") or {}
print("\n  callig_map 条数 =", len(cm))
for k, v in list(cm.items())[:4]:
    print(f"    {k!r} -> {v!r}")

p2c = m.get("pair_to_callig") or {}
print("\n  pair_to_callig 条数 =", len(p2c))
for k, v in list(p2c.items())[:6]:
    print(f"    {k!r} -> {v!r}")
