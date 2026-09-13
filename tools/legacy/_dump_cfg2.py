import sys, json
d = json.load(sys.stdin)
for k in ["vae","vae_path","vae_downscale","latent_channels","vae_scaling_factor",
          "char_embed_dim","data_csv","img_root","condition_fusion","model","use_lora",
          "num_calligraphers","num_characters","max_steps","lr","global_batch_size",
          "eval_csv","eval_n"]:
    print(k, "=", d.get(k))
# 数据规模
import csv
csv_path = d.get("data_csv","")
if csv_path:
    import os
    full = csv_path if os.path.isabs(csv_path) else os.path.join("/root/Workspace/xy/DiT", csv_path)
    try:
        n = sum(1 for _ in open(full, encoding="utf-8")) - 1
        print("csv_rows", "=", n)
    except Exception as e:
        print("csv_rows", "= ERR", e)
# 唯一书家/字符数
try:
    import os
    full = csv_path if os.path.isabs(csv_path) else os.path.join("/root/Workspace/xy/DiT", csv_path)
    rows = list(csv.DictReader(open(full, encoding="utf-8")))
    print("unique_calligraphers", "=", len(set(r["calligrapher_id"] for r in rows)))
    print("unique_chars", "=", len(set(r["character_id"] for r in rows)))
    print("unique_scripts", "=", len(set(r["script_id"] for r in rows)))
except Exception as e:
    print("stats ERR", e)
