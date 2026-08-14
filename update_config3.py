import json, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
p = "train_full_3cond.json"
c = json.load(open(p, encoding="utf-8"))
# 数据/模式
c["data_csv"] = "final_train.csv"
c["eval_csv"] = "final_eval.csv"
c["data_dir"] = ""
c["latent_shards_dir"] = "final_latents"
# 关闭 canny/skel
c["use_canny"] = False
c["use_skel"] = False
# save / autoeval = 1000
c["ckpt_every"] = 1000
c["auto_eval"] = True
c["eval_n"] = 1000
# 结果目录
c["results_dir"] = "results/full_3cond"
json.dump(c, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("updated:", {k: c.get(k) for k in ["data_csv","eval_csv","data_dir","latent_shards_dir","use_canny","use_skel","ckpt_every","auto_eval","eval_n","results_dir","num_calligraphers","num_scripts","num_characters"]})
