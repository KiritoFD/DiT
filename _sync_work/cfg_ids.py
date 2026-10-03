import json, os
os.chdir("/root/Workspace/xy/DiT")
d = json.load(open("src/train/configs/v12_pretrain_S_cat_fame_kxl_tj_px60.json", encoding="utf-8"))
for k in ["callig_id_map", "num_calligraphers", "num_characters", "data_csv",
          "latent_shards_dir", "skel_latent_shards_dir", "no_char_cond", "use_char_cond"]:
    print("  %-24s %s" % (k, d.get(k, "<ABSENT>")))

for f in ["assets/callig_id_map.json", "assets/callig_id_map_base.json", "assets/callig_id_map_tj.json"]:
    m = json.load(open(f, encoding="utf-8"))
    print("%-36s num=%s len=%s" % (f, m.get("num_calligraphers"), len(m.get("id_map", {}))))

# 我们的 csv 用到的 calligrapher_id 是否都被 map 覆盖
import csv
rows = list(csv.DictReader(open("assets/train_fame-kxl-tj-px60.csv", encoding="utf-8")))
used = {int(r["calligrapher_id"]) for r in rows}
for f in ["assets/callig_id_map.json", "assets/callig_id_map_base.json"]:
    m = json.load(open(f, encoding="utf-8"))["id_map"]
    keys = {int(k) for k in m}
    print(f"{f}: 覆盖我们 csv 的 {len(used & keys)}/{len(used)}; 未覆盖 {sorted(used - keys)[:10]}")

# 模型侧 num_calligraphers 的语义
import re
src = open("src/train/train.py", encoding="utf-8").read()
i = src.find("load_callig_id_map")
print("\n--- train.py 加载处 ---")
print(src[i - 300:i + 500])
