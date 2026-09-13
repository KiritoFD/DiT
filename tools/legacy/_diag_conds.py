"""Check conds alignment: does eval cache use the right (callig_id, glyph_id) for each index?"""
import os, sys, csv, json
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir("/root/Workspace/xy/DiT")

rows = list(csv.DictReader(open("5script/eval500_clean.csv", encoding="utf-8")))
print("First 10 eval CSV rows:")
for i, r in enumerate(rows[:10]):
    print(f"  [{i}] callig={r['calligrapher_id']} glyph={r['glyph_id']} char={r['character']!r} script={r['script']!r} img={r['image_path']!r}")

# Check the conds: our in_process_eval uses row["calligrapher_id"] and row.get("glyph_id", row.get("character_id",0))
print("\nCond extraction (from in_process_eval logic):")
for i, r in enumerate(rows[:10]):
    callig_id = int(r["calligrapher_id"])
    glyph_id = int(r.get("glyph_id", r.get("character_id", 0)))
    print(f"  [{i}] callig_id={callig_id} glyph_id={glyph_id}")
