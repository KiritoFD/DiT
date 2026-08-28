import json, sys, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
p = "train_full_3cond.json"
c = json.load(open(p, encoding="utf-8"))
c["results_dir"] = "new_data/results_full_3cond"
json.dump(c, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
os.makedirs("new_data", exist_ok=True)
print("results_dir:", c["results_dir"], "global_batch_size:", c.get("global_batch_size"))
