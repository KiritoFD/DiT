#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""build_seen_control.py — "可直写性"的上界对照：拿一个**已经训过**的书家当对照。

问题: row_pt(零梯度把 50 张图的 DINO 质心写进新行) 已经比 mean_scaled 好，
      但它离"这一行被 150k 步真训出来"还差多少？
做法: 选一个在 87-pair 表里的 (书家×书体)，例如 米芾-行：
      S2 = 用**真实训好的那一行**在同一批留出字上评测（上界，模型见过这些字所属书家）
      S1 = 把该字 50 张图算出的质心**手写**进一个全新的第 88 行，零梯度评测
      S1b= 同一个新行但用 87 行均值初始化（下界）
      若 S1 ≈ S2，则"风格条件装得下全新书家"不只是定性成立，而是**表示可直接写入**；
      若 S1 明显低于 S2，差额就是 150k 步梯度买到的东西 —— 那才是 few-shot 训练的目标区间。

用法:
    /opt/conda/envs/cu121/bin/python _sync_work/build_seen_control.py 米芾-行
    bash _sync_work/run_fs6.sh base 米芾-行-ctrl        # S1 / S1b
    # S2 见打印出来的那条 --in-mem-eval-sets 命令
"""
import argparse
import csv
import json
import os
import random
import sys

import numpy as np

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from _sync_work.build_fs_topic import (N_EVAL, N_TRAIN, SEED, ID_BASE,  # noqa: E402
                                       ID_SLOT_STRIDE, encode_shard, dino_row)

CSV50K = "assets/train_50k_v2.csv"
MAP50K = "assets/callig_script_id_map.json"
BASE_CFG = "src/train/configs/v15a_multistyle_k4_pool.json"
NEW_RAW_ID = "9999"
CTRL_SLOT = 20            # 号段与真主题(0..5)分开: 920000+


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("topic", help="形如 米芾-行，必须是 87-pair 表里已有的 pair")
    a = ap.parse_args()
    cal, _, sc = a.topic.partition("-")
    m = json.load(open(MAP50K, encoding="utf-8"))
    rows = list(csv.DictReader(open(CSV50K, encoding="utf-8")))
    id2name = {r["calligrapher_id"]: r["calligrapher"] for r in rows}
    cid_raw = [k for k, v in id2name.items() if v == cal]
    if not cid_raw:
        raise SystemExit(f"50k 里没有书家 {cal}")
    sid = {"楷": "0", "行": "3", "隶": "4"}[sc]
    key = f"{cid_raw[0]}:{sid}"
    if key not in m["pair_map"]:
        raise SystemExit(f"{a.topic} 不在 87-pair 表里（对照要用**已训过**的 pair）")
    real_pair = m["pair_map"][key]

    mine = [r for r in rows if r["calligrapher_id"] == cid_raw[0] and r["script_id"] == sid]
    by_char = {}
    for r in mine:
        by_char.setdefault(r["character"], []).append(r)
    chars = sorted(c for c, v in by_char.items()
                   if all(os.path.isfile(x["image_path"]) and os.path.isfile(x["std_path"])
                          for x in v[:1]))
    need = N_TRAIN + N_EVAL
    if len(chars) < need:
        raise SystemExit(f"{a.topic} 可用字 {len(chars)} < {need}")
    rng = random.Random(SEED + CTRL_SLOT)
    rng.shuffle(chars)
    write_ch, eval_ch = chars[:N_TRAIN], chars[N_TRAIN:N_TRAIN + N_EVAL]
    print(f"[对照] {a.topic} 真实 pair={real_pair}  该 pair 共 {len(mine)} 张 / {len(chars)} 字"
          f" -> 写行用 {len(write_ch)} 字，评测用 {len(eval_ch)} 字（按字互斥）")

    def pick(chs):
        return [by_char[c][0] for c in sorted(chs)]

    w_rows, e_rows = pick(write_ch), pick(eval_ch)

    # ---- S1 的评测集: 同一批图，但 calligrapher_id 换成 9999、img_id 换成独立号段 ----
    id0 = ID_BASE + CTRL_SLOT * ID_SLOT_STRIDE
    allr = w_rows + e_rows
    new_ids = {r["image_path"]: id0 + i for i, r in enumerate(allr)}
    cols = list(e_rows[0].keys())
    if "img_id" not in cols:
        cols = cols + ["img_id"]

    def dump(path, rows_, renumber):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows_:
                r = dict(r)
                if renumber:
                    r["calligrapher_id"] = NEW_RAW_ID
                    r["script_id"] = sid
                    r["img_id"] = str(new_ids[r["image_path"]])
                w.writerow(r)

    p_write = f"assets/fs6c_{a.topic}_write.csv"
    p_eval_new = f"assets/fs6c_{a.topic}_eval_new.csv"
    p_eval_real = f"assets/fs6c_{a.topic}_eval_real.csv"
    dump(p_write, w_rows, True)
    dump(p_eval_new, e_rows, True)
    dump(p_eval_real, e_rows, False)       # S2 用：真实 id，走 50k 现成的 shard

    encode_shard([r["image_path"] for r in allr], [new_ids[r["image_path"]] for r in allr],
                 [os.path.basename(r["image_path"]) for r in allr],
                 f"data/50k/shards_fs6c_{a.topic}", "img")
    encode_shard([r["std_path"] for r in allr], [new_ids[r["image_path"]] for r in allr],
                 [os.path.basename(r["std_path"]) for r in allr],
                 f"data/50k/shards_fs6c_{a.topic}_std", "std")
    dino_row([r["image_path"] for r in w_rows], f"assets/fs6c_row_{a.topic}.pt")

    # ---- pair map（加第 88 行）与配置 ----
    mm = json.loads(json.dumps(m))
    new_pair = int(mm["num_pairs"])
    mm["pair_map"][f"{NEW_RAW_ID}:{sid}"] = new_pair
    mm["callig_map"][NEW_RAW_ID] = int(mm["num_calligraphers"])
    mm["num_pairs"] = new_pair + 1
    mm["num_calligraphers"] = int(mm["num_calligraphers"]) + 1
    if isinstance(mm.get("pair_to_callig"), list):
        mm["pair_to_callig"].append(int(mm["num_calligraphers"]) - 1)
    p_map = f"assets/callig_script_id_map_fs6c_{a.topic}.json"
    json.dump(mm, open(p_map, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    d = json.loads(json.dumps(json.load(open(BASE_CFG, encoding="utf-8"))))
    d.update({
        "experiment_name": f"v15-fs6c-{a.topic}",
        "results_dir": f"assets/results/v15_fs6c_{a.topic}",
        "num_calligraphers": new_pair + 1,
        "callig_script_map": p_map,
        "init_new_callig": "row_pt",
        "new_callig_pt": f"assets/fs6c_row_{a.topic}.pt",
        "data_csv": p_write,
        "in_mem_eval_sets": f"fewshot:{p_eval_new}:{N_EVAL}",
        "latent_shards_dir": f"data/50k/shards_fs6c_{a.topic}",
        "skel_latent_shards_dir": f"data/50k/shards_fs6c_{a.topic}_std",
        "eval_skel_latent_shards_dir": f"data/50k/shards_fs6c_{a.topic}_std",
        "w_repa": 0.0, "weight_decay": 0.0, "use_ema": False,
        "style_anchor_weight": 0.0, "cond_drop_all_prob": 0.0,
        "global_batch_size": N_TRAIN, "max_steps": 150000 + 1000,
        "ckpt_every": 250, "epoch_steps": 250, "gpu_eval_every": 250,
        "lr": 1e-3, "lr_schedule": "constant", "warmup_steps": 50,
    })
    p_cfg = f"src/train/configs/v15_fs6c_{a.topic}.json"
    json.dump(d, open(p_cfg, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print(f"""
[产出] {p_cfg}
跑三个臂（都是 --eval-only，零训练）:
  S1  手写的第 88 行(row_pt):
    $PY -u src/train/train.py --config {p_cfg} --resume-full $CK --eval-only \\
        --init-new-callig row_pt --results-dir /tmp/_fs6c_{a.topic}_row_pt
  S1b 均值行(mean_scaled, 下界):  同上换 --init-new-callig mean_scaled
  S2  真实训好的第 {real_pair} 行(上界, 用 50k 现成 shard):
    $PY -u src/train/train.py --config {BASE_CFG} --resume-full $CK --eval-only \\
        --use-ema false --in-mem-eval-sets "ctrl:{p_eval_real}:{N_EVAL}" \\
        --results-dir /tmp/_fs6c_{a.topic}_trained_row
判读: S1 距 S2 的差 = 150k 步梯度买到的量；S1 距 S1b 的差 = "写法"买到的量。
""")


if __name__ == "__main__":
    main()
