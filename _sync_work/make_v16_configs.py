#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""make_v16_configs.py — 生成 v16 系列 few-shot 配置。

v16（用户裁定）：**串行** + **和预训练同款的 cosine 衰减**：
  * 峰值 lr **1e-4**（与预训练一致；1e-3 试过一轮被否掉，太高）
  * cosine 衰减到 1e-5（min_lr_ratio=0.1，同预训练），warmup 500
  * max_steps=50000 且配合 `--fresh-scheduler`：新行参数与预训练参数**不重合**，
    所以不继承优化器状态/步数计数器，LR 调度从 step 0 重新走完整条 cosine。
  * batch 384，每 2500 步 eval
"""
import json
import os

TOPICS = ["沈周-行", "伊秉绶-行", "傅山-行", "伊秉绶-隶", "徐渭-行",
          "张即之-楷", "郑道昭-楷", "张裕钊-楷"]
PATCH = {
    "lr": 1e-4,
    "lr_schedule": "cosine",
    "min_lr_ratio": 0.1,          # 1e-4 * 0.1 = 1e-5 收尾（与预训练同款）
    "warmup_steps": 500,
    "max_steps": 50000,           # 配合 --fresh-scheduler：整条 cosine 就是这 50k 步
    "ckpt_every": 5000,
    "epoch_steps": 5000,
    "gpu_eval_every": 2500,
    "global_batch_size": 384,
}

os.chdir("/root/Workspace/xy/DiT")
for t in TOPICS:
    src = f"src/train/configs/v15_fs6_{t}.json"
    if not os.path.exists(src):
        print(f"[skip] 缺 {src}")
        continue
    d = json.load(open(src, encoding="utf-8"))
    d.update(PATCH)
    d["experiment_name"] = f"v16-fs-{t}"
    d["results_dir"] = f"assets/results/v16_fs_{t}"
    out = f"src/train/configs/v16_fs_{t}.json"
    json.dump(d, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"[v16] {out}  lr={d['lr']} batch={d['global_batch_size']} "
          f"steps={d['max_steps']} eval@{d['gpu_eval_every']}")
