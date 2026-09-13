import json, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
p = "train_full_3cond.json"
c = json.load(open(p, encoding="utf-8"))
c["num_calligraphers"] = 1873
c["num_scripts"] = 12
c["num_characters"] = 7765
c["data_csv"] = "final_train.csv"
c["eval_csv"] = "final_eval.csv"
c["data_dir"] = ""
json.dump(c, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("updated:", {k: c[k] for k in ["num_calligraphers","num_scripts","num_characters","data_csv","eval_csv","data_dir"]})
