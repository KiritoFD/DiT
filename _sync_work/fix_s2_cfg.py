#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""调整 v17_s2_* 训练配置：
  - max_steps: 250000 -> 40000   （依据 v13_12ch_post 22.5k 就最好；15c 210k 最差）
  - lr:        0.0001 -> 0.0005  （新通路 zero-init，梯度信号弱，需更高 lr）
  - 打印 diff 以便复核
"""
import json
import glob
import os

NEW_STEPS = 40000
NEW_LR = 0.0005
CKPT_EVERY = 5000

files = sorted(glob.glob("/root/Workspace/xy/DiT/src/train/configs/v17_s2_s2*.json"))
for f in files:
    with open(f, encoding="utf-8") as fh:
        c = json.load(fh)
    old_steps = c.get("max_steps")
    old_lr = c.get("lr")
    c["max_steps"] = NEW_STEPS
    c["lr"] = NEW_LR
    c["ckpt_every"] = CKPT_EVERY
    with open(f, "w", encoding="utf-8") as fh:
        json.dump(c, fh, ensure_ascii=False, indent=2)
    print("%-42s steps %s->%s   lr %s->%s"
          % (os.path.basename(f), old_steps, NEW_STEPS, old_lr, NEW_LR))
print("\ndone: %d files" % len(files))
