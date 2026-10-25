#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/run_serial_decoupled_114m.py — 串行运行 114M 条件解耦与 STD 增强两组消融实验 (70k 步预算):
  1. Exp A: Writer-Decoupled (114M + 条件彻底解耦 + 分尺度 tanh Writer FiLM)
     -> 从 50k 检查点平滑断点续训至 70,000 步
  2. Exp B: Writer-Decoupled + STD (Exp A + Decoder 16²/32² STD 结构直接残差)
     -> 全量 70,000 步训练
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/root/Workspace/xy/DiT")
PYTHON = "/opt/conda/envs/cu121/bin/python"
QUEUE_DIR = ROOT / "scratch/unet_decoupled_114m"

JOBS = [
    {
        "name": "armA_writer_decoupled_70k",
        "desc": "Exp A: 114M 条件解耦 + 分尺度限幅 FiLM (续训 50k -> 70k)",
        "config": ROOT / "src/unet/configs/decoupled_114m/armA_writer_decoupled_70k.json",
        "resume": ROOT / "assets/results/unet_decoupled_114m/armA_writer_decoupled_50k/20261019-191851-Unet-flow/checkpoints/0050000.pt",
        "output_dir": ROOT / "assets/results/unet_decoupled_114m/armA_writer_decoupled_70k",
        "target_step": "0070000.pt",
    },
    {
        "name": "armB_writer_decoupled_std_70k",
        "desc": "Exp B: Exp A + Decoder 16²/32² STD 结构直接残差 (全新 70k 训练)",
        "config": ROOT / "src/unet/configs/decoupled_114m/armB_writer_decoupled_std_70k.json",
        "resume": None,
        "output_dir": ROOT / "assets/results/unet_decoupled_114m/armB_writer_decoupled_std_70k",
        "target_step": "0070000.pt",
    },
]


def write_status(phase: str, **fields):
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    record = dict(phase=phase, time=now, **fields)
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    target = QUEUE_DIR / "status.json"
    temp = target.with_suffix(".tmp")
    temp.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(target)
    print(f"[{now}] status={phase} {json.dumps(fields, ensure_ascii=False)}", flush=True)


def is_job_done(output_dir: Path, target_step: str) -> bool:
    if not output_dir.exists():
        return False
    return any(p.name == target_step for p in output_dir.glob("*/checkpoints/" + target_step))


def main():
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    write_status(
        "queue_started",
        total_jobs=len(JOBS),
        jobs=[j["name"] for j in JOBS],
    )

    for idx, job in enumerate(JOBS):
        name = job["name"]
        desc = job["desc"]
        cfg = job["config"]
        resume = job.get("resume")
        out_dir = job["output_dir"]
        target = job["target_step"]
        log_file = QUEUE_DIR / f"{name}.log"

        if is_job_done(out_dir, target):
            print(f"[{name}] Checkpoint {target} already exists in {out_dir}, skipping...", flush=True)
            write_status("job_skipped", current_job=name, index=idx + 1, total=len(JOBS), reason="checkpoint_exists")
            continue

        cmd = [
            PYTHON, "-u", "-m", "src.unet.train",
            "--config", str(cfg),
        ]
        if resume:
            cmd.extend(["--resume", str(resume)])

        print("=" * 70, flush=True)
        print(f"Starting Job {idx+1}/{len(JOBS)}: {name} ({desc})", flush=True)
        print(f"Command: {' '.join(cmd)}", flush=True)
        print(f"Log: {log_file}", flush=True)
        print("=" * 70, flush=True)

        start_time = time.time()
        with open(log_file, "a", encoding="utf-8") as out:
            proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                env=dict(os.environ, PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1"),
                stdout=out,
                stderr=subprocess.STDOUT,
            )
            write_status("running", current_job=name, desc=desc, index=idx + 1, total=len(JOBS), pid=proc.pid)
            exit_code = proc.wait()

        elapsed = time.time() - start_time
        if exit_code != 0:
            write_status("failed", current_job=name, index=idx + 1, exit_code=exit_code, elapsed_s=elapsed)
            print(f"Job {name} FAILED with exit code {exit_code}! Aborting queue.", flush=True)
            sys.exit(exit_code)

        write_status("job_completed", current_job=name, index=idx + 1, total=len(JOBS), elapsed_s=elapsed)
        print(f"Job {name} COMPLETED successfully in {elapsed/3600:.2f} hours!", flush=True)

    write_status("all_completed", total_jobs=len(JOBS))
    print("All 114M decoupled 70k jobs completed successfully!", flush=True)


if __name__ == "__main__":
    main()
