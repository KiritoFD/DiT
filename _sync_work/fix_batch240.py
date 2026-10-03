import json
import os

os.chdir("/root/Workspace/xy/DiT")
p = "src/train/configs/v15c_fixed.json"
d = json.load(open(p, encoding="utf-8"))
d["global_batch_size"] = 240        # 320 OOM，回到 240
d["compile_mode"] = "reduce-overhead"
json.dump(d, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("  batch =", d["global_batch_size"])
print("  compile_mode =", d["compile_mode"])
print("  ckpt_every =", d["ckpt_every"])

# 查 703:4 是什么
import csv
for f in ("assets/eval_v13_strict_fixed.csv", "assets/eval_v13_seen_fixed.csv"):
    if not os.path.exists(f):
        continue
    for r in csv.DictReader(open(f, encoding="utf-8")):
        if r.get("calligrapher_id") == "703" and r.get("script_id") == "4":
            print(f"  {os.path.basename(f)}: 703:4 -> "
                  f"{r['calligrapher']}/{r['script']}/{r['character']}")
            break
