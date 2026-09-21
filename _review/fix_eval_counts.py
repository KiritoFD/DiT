import csv
import json
import os

os.chdir("/root/Workspace/xy/DiT")
p = "src/train/configs/v15c_fixed.json"
d = json.load(open(p, encoding="utf-8"))

# 按实际行数重写 in_mem_eval_sets
sets = []
for name, f in (("seen", "assets/eval_v13_seen_fixed.csv"),
                ("strict", "assets/eval_v13_strict_fixed.csv")):
    n = len(list(csv.DictReader(open(f, encoding="utf-8"))))
    sets.append(f"{name}:{f}:{n}")
d["in_mem_eval_sets"] = ",".join(sets)
d["global_batch_size"] = 240
json.dump(d, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

print("  in_mem_eval_sets =", d["in_mem_eval_sets"])
print("  global_batch_size =", d["global_batch_size"])
print("  compile_mode =", d["compile_mode"])
print("  ckpt_every =", d["ckpt_every"])
