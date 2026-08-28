import json, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
p = "train_full_3cond.json"
c = json.load(open(p, encoding="utf-8"))
c["use_canny"] = True
c["use_skel"] = False
c["img_root"] = "final_imgs_256"
c["canny_root"] = "final_canny"
json.dump(c, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("updated:", {k: c.get(k) for k in ["use_canny","use_skel","img_root","canny_root","ckpt_every","auto_eval","eval_n","latent_shards_dir"]})
