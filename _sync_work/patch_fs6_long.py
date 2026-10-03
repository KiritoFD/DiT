#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""patch_fs6_long.py — 把 few-shot 配置改成"充分训练"档：+50k 步、稀疏 eval、常量 lr。

判读以 Diff 进入平台期为准（用户要求），所以步数预算 50k、每 2500 步 eval/记录。
"""
import json

PATCH = {
    "lr": 3e-4,
    "lr_schedule": "constant",
    "warmup_steps": 100,
    "max_steps": 200000,          # 150000 + 50000
    "ckpt_every": 5000,
    "epoch_steps": 5000,
    "gpu_eval_every": 2500,
    "global_batch_size": 384,     # 用户要求把显存吃到 ~20G：3 路并行 x 384
}
TOPICS = ["沈周-行", "伊秉绶-行", "傅山-行", "伊秉绶-隶", "徐渭-行"]
for t in TOPICS:
    p = f"src/train/configs/v15_fs6_{t}.json"
    d = json.load(open(p, encoding="utf-8"))
    d.update(PATCH)
    json.dump(d, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"[patch] {p}: lr={d['lr']} max_steps={d['max_steps']} "
          f"eval@{d['gpu_eval_every']} ckpt@{d['ckpt_every']}")
