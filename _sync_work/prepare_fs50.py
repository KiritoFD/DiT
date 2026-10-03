"""K=50 few-shot 全流程准备：编码 latent -> resize 图 -> 同步 img_id -> 生成配置。

顺序不能乱（实测踩过多次）:
  1) 合并 train+eval 成 _all.csv（保证 img_id 全局唯一）
  2) 编码 latent（此时 image_path 还是原图）
  3) resize 原图到 256 存副本，CSV 指向副本
  4) 把 _all 的 img_id 同步回 train/eval
  5) 生成扩展 callig_id_map + config
"""
import csv
import json
import os

os.chdir("/root/Workspace/xy/DiT")

CALS = [("怀素", "草"), ("伊秉绶", "隶"), ("徐渭", "草"), ("沈周", "行")]
RAW_ID, NEW_ID = 9999, 45
K, EVAL_N = 50, 100
STEPS, CKPT = 200, 100


def combine(cal):
    tr = list(csv.DictReader(open(f"assets/fs50_{cal}_train.csv", encoding="utf-8")))
    ev = list(csv.DictReader(open(f"assets/fs50_{cal}_eval.csv", encoding="utf-8")))
    cols = list(tr[0].keys())
    if "img_id" not in cols:
        cols.append("img_id")
    rows = []
    for i, r in enumerate(tr):
        r["img_id"] = str(i)
        r["calligrapher_id"] = str(RAW_ID)
        rows.append(r)
    for j, r in enumerate(ev):
        r["img_id"] = str(len(tr) + j)
        r["calligrapher_id"] = str(RAW_ID)
        rows.append(r)
    with open(f"assets/fs50_{cal}_all.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return len(tr), len(ev)


def sync_ids(cal):
    allr = list(csv.DictReader(open(f"assets/fs50_{cal}_all.csv", encoding="utf-8")))
    idmap = {r["image_path"]: r["img_id"] for r in allr}
    for name in ("train", "eval"):
        p = f"assets/fs50_{cal}_{name}.csv"
        rows = list(csv.DictReader(open(p, encoding="utf-8")))
        cols = list(rows[0].keys())
        if "img_id" not in cols:
            cols.append("img_id")
        for r in rows:
            r["img_id"] = idmap.get(r["image_path"], "")
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)


def make_idmap(cal):
    base = json.load(open("assets/callig_id_map_50k.json", encoding="utf-8"))
    mm = {"num_calligraphers": 46, "id_map": dict(base["id_map"])}
    mm["id_map"][str(RAW_ID)] = NEW_ID
    out = f"assets/callig_id_map_50k_fs50_{cal}.json"
    json.dump(mm, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return out


def make_config(cal, idmap):
    d = json.load(open("src/train/configs/v13_fs_怀素.json", encoding="utf-8"))
    d["data_csv"] = f"assets/fs50_{cal}_train.csv"
    d["in_mem_eval_sets"] = f"fewshot:assets/fs50_{cal}_eval.csv:{EVAL_N}"
    d["global_batch_size"] = K
    d["num_calligraphers"] = 46
    d["latent_shards_dir"] = f"data/50k/shards_fs50_{cal}"
    d["callig_id_map"] = idmap
    d["use_ema"] = False                 # ★ EMA 会掩盖短跑的更新
    d["lr"] = 1e-4
    d["lr_schedule"] = "constant"
    d["warmup"] = 0
    d["init_new_callig"] = "mean_scaled"  # ★ 均值方向 + 典型书家范数
    d["max_steps"] = 155000 + STEPS
    d["ckpt_every"] = CKPT
    d["epoch_steps"] = CKPT               # 必须与 ckpt_every 相等
    d["freeze_callig_table"] = False
    d["cond_drop_all_prob"] = 0.0
    out = f"src/train/configs/v13_fs50_{cal}.json"
    json.dump(d, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return out


if __name__ == "__main__":
    import sys
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    for cal, script in CALS:
        if step in ("all", "combine"):
            n_tr, n_ev = combine(cal)
            print(f"[{cal}] 合并 train {n_tr} + eval {n_ev}")
        if step in ("all", "sync"):
            sync_ids(cal)
            print(f"[{cal}] img_id 已同步")
        if step in ("all", "cfg"):
            idmap = make_idmap(cal)
            cfg = make_config(cal, idmap)
            print(f"[{cal}] {cfg}")
