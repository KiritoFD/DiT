#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""re_eval.py — 对**已训练**的 checkpoint 单独重跑一次评测（不改动训练循环）。

动机 (2026-10-09)
-----------------
v71 (Calli-VAE) 的 in-mem eval 用的是错的 VAE 解码约定
（``vae.decode(lat/sf)`` == ``decode(mode)``，应为 ``decode(sample/sf)``），
导致所有已存 eval 图/指标都是**灰图** (SSIM≈0.51, MSE≈0.79)，完全不能用。

本脚本复用仓库里已经修好的 in-mem eval 路径（``--eval-only`` + config 里的
``calli_decode_noise: true``），对指定 ckpt 重新算指标并落盘到
``<results_dir>/<timestamp>-<exp>/eval_stdskel_summary.csv``。

用法
----
    # 指定 ckpt
    python tools/re_eval.py --config src/train/configs/v71_callivae_sp_c2ot.json \
        --ckpt assets/results/.../checkpoints/0030000.pt

    # 或让它自动找 results_dir 里最新 ckpt
    python tools/re_eval.py --config src/train/configs/v71_callivae_sp_c2ot.json --latest

    # 后台挂起（写日志）
    python tools/re_eval.py --config ... --latest --detach --log /tmp/reeval.log
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys


def _config_get(cfg, key, default=None):
    if isinstance(cfg, dict):
        v = cfg.get(key, default)
    else:
        v = default
    return v


def resolve_ckpt(a, cfg):
    if a.ckpt:
        return a.ckpt
    # results_dir / experiment_name 从 config 读
    rd = a.results_dir or _config_get(cfg, "results_dir", "")
    exp = _config_get(cfg, "experiment_name", "")
    if not rd:
        raise SystemExit("[re_eval] 无法确定 results_dir，请显式给 --ckpt")
    # 取 results_dir 下最新的运行目录
    runs = sorted(glob.glob(os.path.join(rd, "*")), key=os.path.getmtime, reverse=True)
    runs = [r for r in runs if os.path.isdir(r)]
    if not runs:
        raise SystemExit(f"[re_eval] {rd} 下没有运行目录")
    run = runs[0]
    ckdir = os.path.join(run, "checkpoints")
    cks = [f for f in glob.glob(os.path.join(ckdir, "*.pt")) if not f.endswith(".done")]
    if a.step is not None:
        cands = [f for f in cks if int(re.findall(r"(\d+)", os.path.basename(f))[-1]) == a.step]
        if cands:
            cks = cands
    if not cks:
        raise SystemExit(f"[re_eval] {ckdir} 下没找到 .pt")
    cks.sort(key=lambda f: int(re.findall(r"(\d+)", os.path.basename(f))[-1]))
    print(f"[re_eval] auto-picked ckpt: {cks[-1]}")
    return cks[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default="", help="显式 ckpt 路径；不给则配合 --results-dir/--latest 自动找")
    ap.add_argument("--results-dir", default="", help="覆盖 config 里的 results_dir")
    ap.add_argument("--step", type=int, default=None, help="只挑这个步数的 ckpt")
    ap.add_argument("--latest", action="store_true", help="自动取最新 ckpt")
    ap.add_argument("--sets", default="", help="覆盖 in_mem_eval_sets，如 'eval200fix:...:187,seen:...:20'")
    ap.add_argument("--log", default="", help="--detach 时的日志路径 (默认 /tmp/re_eval.log)")
    ap.add_argument("--python", default=sys.executable or "/home/ds/miniconda3/envs/pytorch/bin/python")
    ap.add_argument("--detach", action="store_true", help="后台运行 (setsid)")
    a = ap.parse_args()

    with open(a.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if a.results_dir:
        cfg["results_dir"] = a.results_dir

    ckpt = resolve_ckpt(a, cfg)
    if not os.path.exists(ckpt):
        raise SystemExit(f"[re_eval] ckpt 不存在: {ckpt}")

    cmd = [a.python, "-u", "-m", "src.train.train",
           "--config", a.config,
           "--resume-full", ckpt,
           "--eval-only"]
    if a.sets:
        cmd += ["--in-mem-eval-sets", a.sets]

    # 保证 Calli-VAE 的 decode 补噪开关是开的（config 已是 true；这里再兜底显式传）
    if bool(cfg.get("calli_decode_noise", False)):
        cmd += ["--calli-decode-noise", "true"]

    print("[re_eval] cmd:", " ".join(cmd), flush=True)
    if a.detach:
        log = a.log or "/tmp/re_eval.log"
        with open(log, "w", encoding="utf-8") as lf:
            p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                 start_new_session=True, cwd=os.getcwd())
        print(f"[re_eval] started pid={p.pid}, log={log}")
    else:
        subprocess.run(cmd, cwd=os.getcwd())


if __name__ == "__main__":
    main()
