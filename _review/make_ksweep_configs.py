#!/usr/bin/env python
"""生成 K 扫描的 few-shot 配置（K = 5 / 10 / 20 / 50）。

每个 K 一份 config，data_csv 指向对应的 train csv。
"""
import json
import os

os.chdir("/root/Workspace/xy/DiT")
BASE = "src/train/configs/v13_fewshot_k10.json"
d0 = json.load(open(BASE, encoding="utf-8"))

for K in (5, 10, 20, 50):
    d = dict(d0)
    d["data_csv"] = f"assets/fewshot_huaisu_k{K}_train.csv"
    d["in_mem_eval_sets"] = f"fewshot:assets/fewshot_huaisu_k{K}_eval.csv:100"
    d["global_batch_size"] = K
    d["num_calligraphers"] = 46
    out = f"src/train/configs/v13_fewshot_k{K}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
    print(f"  {out}  (k={K}, batch={K})")
print("\n  数据用: python tools/build_fewshot_csv.py --cal 怀素 --script 草 "
      "--k <K> --eval-n 100 --out-prefix assets/fewshot_huaisu_k<K>")
