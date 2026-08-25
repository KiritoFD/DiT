#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""一次性: 为 s7 拉取 eval_samples + seen_samples, 生成 s7 poster。
s7 训练时没拉 poster, 这里补上。
"""
import os, sys, subprocess
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pull_monitor as pm

S7_CKPT = "/root/Workspace/xy/DiT/5script/results/s7_klf4_top30/20260823-163710-s7-klf4-top30-diffonly/checkpoints"

# s7 专用本地目录, 和 s8 分开
S7_ES = os.path.join(HERE, "s7_eval_samples")
S7_SEEN = os.path.join(HERE, "s7_seen_samples")
S7_POSTER = os.path.join(HERE, "s7_eval_poster.png")

print("=== pulling s7 eval_samples ===")
# 用 pull_monitor 的 _scp_dir 但指定本地目录
os.makedirs(S7_ES, exist_ok=True)
pm._scp_dir(f"{S7_CKPT}/eval_samples/", S7_ES, timeout=300)
# 展平嵌套
import shutil
nested = os.path.join(S7_ES, "eval_samples")
if os.path.isdir(nested):
    for d in os.listdir(nested):
        src = os.path.join(nested, d)
        dst = os.path.join(S7_ES, d)
        if os.path.isdir(src) and not os.path.exists(dst):
            shutil.move(src, dst)
    shutil.rmtree(nested, ignore_errors=True)

print("=== pulling s7 seen_samples ===")
os.makedirs(S7_SEEN, exist_ok=True)
pm._scp_dir(f"{S7_CKPT}/seen_samples/", S7_SEEN, timeout=300)
nested = os.path.join(S7_SEEN, "seen_samples")
if os.path.isdir(nested):
    for d in os.listdir(nested):
        src = os.path.join(nested, d)
        dst = os.path.join(S7_SEEN, d)
        if os.path.isdir(src) and not os.path.exists(dst):
            shutil.move(src, dst)
    shutil.rmtree(nested, ignore_errors=True)

# 统计
es_steps = pm._existing_steps(S7_ES)
seen_steps = pm._existing_steps(S7_SEEN)
print(f"s7 eval_samples: {len(es_steps)} steps, s7 seen_samples: {len(seen_steps)} steps")
print(f"  eval steps: {es_steps}")
print(f"  seen steps: {seen_steps}")

# 生成 poster
if not os.path.exists(os.path.join(HERE, "make_eval_poster.py")):
    print("make_eval_poster.py not found, skip poster")
    sys.exit(1)

show5_steps = [s for s in es_steps
               if os.path.exists(os.path.join(S7_ES, "step%07d" % s, "samples.json"))]
seen5_steps = [s for s in seen_steps
               if os.path.exists(os.path.join(S7_SEEN, "step%07d" % s, "samples.json"))]
print(f"complete (with samples.json): show5={len(show5_steps)} seen5={len(seen5_steps)}")

if not show5_steps:
    print("no complete show5 steps, skip poster")
    sys.exit(1)

args = [sys.executable, os.path.join(HERE, "make_eval_poster.py"),
        "--show5-dir", S7_ES,
        "--gt-dir", os.path.join(HERE, "remote_gt"),
        "--show5-csv", os.path.join(HERE, "show5_top30.csv"),
        "--eval-json-dir", S7_CKPT,
        "-o", S7_POSTER]
if seen5_steps:
    args.extend(["--seen5-dir", S7_SEEN,
                 "--seen5-csv", os.path.join(HERE, "seen5_top30.csv")])

print("=== generating s7 poster ===")
r = subprocess.run(args, capture_output=True, text=True, timeout=180)
print("stdout:", r.stdout[-500:] if r.stdout else "(none)")
if r.stderr:
    print("stderr:", r.stderr[-500:])
print(f"\ns7 poster -> {S7_POSTER}")
print(f"  exists: {os.path.exists(S7_POSTER)}")
if os.path.exists(S7_POSTER):
    print(f"  size: {os.path.getsize(S7_POSTER)} bytes")