# -*- coding: utf-8 -*-
"""remote: check MCCD id map for zhao mengfu + tongji calligraphers; dataset writers."""
import json

m = json.load(open("labels/final_id_maps.json", encoding="utf-8"))
c2n = m["id_to_name"]["calligrapher"]
names = {v: k for k, v in c2n.items()}
target = ["赵孟頫", "趙孟頫", "赵孟俯", "王宠", "白蕉", "梁诗正", "谭延闿", "文征明",
          "文徵明", "林散之", "吴让之", "姜夔", "柯九思", "顾仲安", "沙孟海", "弘一"]
out = open("mccd_target_check.txt", "w", encoding="utf-8")
for t in target:
    out.write(f"{t}: {names.get(t, 'NOT FOUND')}\n")
# fuzzy: any name containing zhao/meng
out.write("\nnames containing 孟: ")
out.write(", ".join(f"{v}(id {k})" for k, v in c2n.items() if "孟" in v) + "\n")
out.write(f"\nMCCD total calligraphers in map: {len(c2n)}\n")
# dataset actual writers
import os
w = sorted(os.listdir("dataset/images"))
out.write(f"dataset/images writers ({len(w)}): {w}\n")
out.write("their names: " + ", ".join(f"{w_}->{c2n.get(w_, '?')}" for w_ in w) + "\n")
# count images per writer
for w_ in w:
    n = len([x for x in os.listdir(os.path.join("dataset/images", w_))])
    out.write(f"  {w_} ({c2n.get(w_, '?')}): {n} imgs\n")
out.close()
print("done")
