#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""清空 cpu_eval_state.json 里的 error 条目，让 eval 重新尝试。"""
import json, os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

p = "/root/Workspace/xy/DiT/5script/results/s8_klf4_clean_dino/20260823-234546-s8-klf4-clean-dino/checkpoints/cpu_eval_state.json"
state = json.load(open(p, encoding="utf-8"))
print("before:", len(state), "entries")
# 删除所有带 error 的条目
cleaned = {k: v for k, v in state.items() if "error" not in v}
print("after:", len(cleaned), "entries")
json.dump(cleaned, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("done")