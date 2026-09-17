# -*- coding: utf-8 -*-
"""打印 ckpt 的 resume 关键状态 (用于确认 --resume-full 会不会踩 step 推断坑)。

用法: /opt/conda/envs/cu121/bin/python _review/ckpt_info.py <ckpt.pt> [...]
刻意用 io.open + utf-8, 避免 Windows 侧改文件编码。
"""
import io
import os
import re
import sys

import torch

for p in sys.argv[1:]:
    print("=" * 78)
    print("FILE:", p)
    if not os.path.exists(p):
        print("  !! NOT FOUND")
        continue
    print("  size: %.1f MB" % (os.path.getsize(p) / 1e6))
    ck = torch.load(p, map_location="cpu", weights_only=False)
    print("  top-level keys:", sorted(ck.keys()))

    _fname = os.path.basename(p)
    _digits = re.findall(r"\d+", _fname)
    print("  filename digits=%s -> train.py 会推断 resume_start_step=%s"
          % (_digits, int(_digits[-1]) if _digits else "N/A(保留0)"))

    for k in ("train_steps", "step", "global_step", "epoch"):
        if k in ck:
            print("  ck[%r] = %r" % (k, ck[k]))
    a = ck.get("args", None)
    if a is not None:
        ts = getattr(a, "train_steps", None)
        print("  ck['args'].train_steps = %r  <-- 若非 None, 会**覆盖**文件名推断" % (ts,))
        for k in ("model", "global_batch_size", "lr", "max_steps", "warmup_steps",
                  "lr_schedule", "min_lr_ratio", "seed", "global_seed",
                  "skel_as_glyph_cond", "glyph_inject_mode", "glyph_vec_cond"):
            if hasattr(a, k):
                print("     args.%s = %r" % (k, getattr(a, k)))
    for k in ("delta", "opt", "ema", "model", "scheduler", "lr_sched"):
        v = ck.get(k, None)
        if v is None:
            print("  [%s] ABSENT" % k)
        elif isinstance(v, dict):
            print("  [%s] dict, %d entries" % (k, len(v)))
        else:
            print("  [%s] %s" % (k, type(v).__name__))

    # train.py:504 用 `_rf.get("delta", ...)` 当模型权重 -> 它必须是完整 state_dict。
    d = ck.get("delta", None)
    if isinstance(d, dict):
        ks = list(d.keys())
        _orig = sum(1 for k in ks if k.startswith("_orig_mod."))
        print("  delta: %d keys, %d 带 _orig_mod. 前缀 (train.py:506 会剥离)"
              % (len(ks), _orig))
        print("  delta 样例键:", ks[:3], "...", ks[-2:])
        _numel = sum(int(v.numel()) for v in d.values() if hasattr(v, "numel"))
        print("  delta 总元素数: %,d".replace("%,d", "{:,}").format(_numel))
    if isinstance(ck.get("ema"), dict):
        print("  ema 样例键:", list(ck["ema"].keys())[:3])
