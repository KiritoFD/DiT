"""v15 版 few-shot：把新的「书家x书体」pair 注册进 pair_map，并生成配置。

v15 的条件是 **书家x书体 pair**（87 个），每对 4 个子风格 x 384 维 = 1536 维。
这比 v13 的「45 书家 x 128 维单向量」容量大 12 倍，正好用来验证
"更多风格容量能否 few-shot 捕获新书家"。

做法:
  1) 给新书家分配一个未使用的 callig_id (9999)
  2) pair_map 加 "9999:<script_id>" -> 87（新的 pair 槽位）
  3) num_pairs 87->88, num_calligraphers 45->46
  4) pair_to_callig 追加新 callig 的 index (45)
  5) CSV 的 calligrapher_id = 9999, script_id = 该书体的 id
  6) 配置: num_calligraphers=88, 基模用 v15a 的 ckpt
"""
import csv
import json
import os

os.chdir("/root/Workspace/xy/DiT")
SRC_MAP = "assets/callig_script_id_map.json"
V15_CFG = "src/train/configs/v15a_multistyle_k4_pool.json"

# (书家, 书体)
CALS = [("伊秉绶", "隶"), ("沈周", "行"), ("徐渭", "行"), ("宋高宗", "楷")]
NEW_CALLIG_ID = "9999"

base = json.load(open(SRC_MAP, encoding="utf-8"))
v15 = json.load(open(V15_CFG, encoding="utf-8"))

# 从 v15 训练数据里拿到 script 名 -> script_id
rows50 = list(csv.DictReader(open(v15["data_csv"], encoding="utf-8")))
script_ids = {}
for r in rows50:
    script_ids.setdefault(r["script"], r["script_id"])
print(f"  训练数据里的书体: {script_ids}")
for _, sc in CALS:
    if sc not in script_ids:
        print(f"  ⚠ 书体 {sc} 不在训练数据里，将用 0")


def build(cal, script):
    sid = script_ids.get(script, "0")
    m = json.loads(json.dumps(base))          # 深拷贝
    pair_key = f"{NEW_CALLIG_ID}:{sid}"
    if pair_key in m["pair_map"]:
        print(f"  ⚠ {cal}: pair {pair_key} 已存在")
        return None
    new_pair = int(m["num_pairs"])
    m["pair_map"][pair_key] = new_pair
    m["callig_map"][NEW_CALLIG_ID] = int(m["num_calligraphers"])
    m["num_pairs"] = new_pair + 1
    m["num_calligraphers"] = int(m["num_calligraphers"]) + 1
    p2c = m.get("pair_to_callig")
    if isinstance(p2c, list):
        p2c.append(int(m["num_calligraphers"]) - 1)
    out = f"assets/callig_script_id_map_fs15_{cal}.json"
    json.dump(m, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # CSV: calligrapher_id / script_id
    for nm in ("train", "eval"):
        p = f"assets/fs50_{cal}_{nm}.csv"
        rs = list(csv.DictReader(open(p, encoding="utf-8")))
        cols = list(rs[0].keys())
        for r in rs:
            r["calligrapher_id"] = NEW_CALLIG_ID
            r["script_id"] = sid
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rs)

    # 配置
    d = json.loads(json.dumps(v15))
    d["callig_id_map"] = out
    d["num_calligraphers"] = new_pair + 1
    d["data_csv"] = f"assets/fs50_{cal}_train.csv"
    d["in_mem_eval_sets"] = f"fewshot:assets/fs50_{cal}_eval.csv:100"
    d["latent_shards_dir"] = f"data/50k/shards_fs50_{cal}"
    d["global_batch_size"] = 50
    d["use_ema"] = False                      # ★ 否则评测用 EMA，短跑看不见更新
    d["lr"] = 1e-4
    d["lr_schedule"] = "constant"
    d["warmup"] = 0
    d["init_new_callig"] = "mean_scaled"
    d["freeze_callig_table"] = False
    d["cond_drop_all_prob"] = 0.0
    d["max_steps"] = 150000 + 200              # v15a ckpt 是 150k
    d["ckpt_every"] = 100
    d["epoch_steps"] = 100                     # 必须相等
    cfg = f"src/train/configs/v15_fs_{cal}.json"
    json.dump(d, open(cfg, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"  {cal}({script}): pair {pair_key} -> {new_pair}, n_pairs={m['num_pairs']}")
    print(f"     {out}")
    print(f"     {cfg}")
    return cfg


if __name__ == "__main__":
    for c, s in CALS:
        build(c, s)
