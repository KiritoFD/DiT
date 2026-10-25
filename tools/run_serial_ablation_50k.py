#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/run_serial_ablation_50k.py — 串行运行两臂 50k 单变量对比消融实验 (Batch 768，吃满显存 ~22G):
  1. Arm 1: 单独加门控 (SiLU + residual_cond_gate=True, b768)
  2. Arm 2: 方案 B: Writer-only 残差调制 (SiLU + writer_film_rank=16 on decoder 16/8, b768)
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
QUEUE_DIR = ROOT / "scratch/unet_ablation_50k"

JOBS = [
    {
        "name": "arm1_resgate_b768",
        "desc": "单独加门控 (residual_cond_gate=True, b768)",
        "config": ROOT / "src/unet/configs/ablation_50k/arm1_resgate_b768.json",
        "ckpt_target": ROOT / "assets/results/unet_ablation_50k/arm1_resgate_b768/checkpoints/0050000.pt",
    },
    {
        "name": "arm2_writer_film_dec_b768",
        "desc": "方案 B: Writer-only 残差调制 (writer_film_rank=16 on dec 16/8, b768)",
        "config": ROOT / "src/unet/configs/ablation_50k/arm2_writer_film_dec_b768.json",
        "ckpt_target": ROOT / "assets/results/unet_ablation_50k/arm2_writer_film_dec_b768/checkpoints/0050000.pt",
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
        ckpt = job["ckpt_target"]
        log_file = QUEUE_DIR / f"{name}.log"

        if ckpt.exists():
            print(f"[{name}] Checkpoint already exists at {ckpt}, skipping...", flush=True)
            write_status("job_skipped", current_job=name, index=idx + 1, total=len(JOBS), reason="checkpoint_exists")
            continue

        cmd = [
            PYTHON, "-u", "-m", "src.unet.train",
            "--config", str(cfg),
        ]

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
    print("All ablation jobs completed successfully!", flush=True)


if __name__ == "__main__":
    main()
