import json, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
p = "train_full_3cond.json"
c = json.load(open(p, encoding="utf-8"))
c["latent_shards_dir"] = "final_latents"
c["use_canny"] = False
c["use_skel"] = False
json.dump(c, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("config updated:", {k: c.get(k) for k in ["latent_shards_dir","use_canny","use_skel","num_calligraphers"]})
