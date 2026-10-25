"""Serial runner for 3 DiC ablation arms warm-starting from 50k checkpoint."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path("/home/ds/Workspace/DiT")
QUEUE_DIR = ROOT / "scratch/unet_dic_3arms_50k_20261018_01"
PYTHON = "/home/ds/miniconda3/envs/vcp-embryo/bin/python"
CKPT = ROOT / "assets/results/unet_connectivity_100000/baseline_resume/20261018-030921-Unet-flow/checkpoints/0050000.pt"

JOBS = [
    {
        "name": "DiC_residual_gate",
        "config": ROOT / "src/unet/configs/connectivity/DiC_residual_gate.json",
    },
    {
        "name": "DiC_stage_cond_adapter",
        "config": ROOT / "src/unet/configs/connectivity/DiC_stage_cond_adapter.json",
    },
    {
        "name": "DiC_stage_cond_independent",
        "config": ROOT / "src/unet/configs/connectivity/DiC_stage_cond_independent.json",
    }
]

def write_status(phase, **fields):
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    record = dict(phase=phase, time=now, **fields)
    target = QUEUE_DIR / "status.json"
    temp = target.with_suffix(".tmp")
    temp.write_text(json.dumps(record, indent=2), encoding="utf-8")
    temp.replace(target)
    print(f"[{now}] {json.dumps(record)}", flush=True)

def main():
    if not CKPT.is_file():
        raise FileNotFoundError(f"Source checkpoint not found: {CKPT}")

    write_status("queue_started", total_jobs=len(JOBS), jobs=[j["name"] for j in JOBS])

    for idx, job in enumerate(JOBS):
        name = job["name"]
        cfg = job["config"]
        log_file = QUEUE_DIR / f"{name}.log"

        cmd = [
            PYTHON, "-u", "-m", "src.unet.train",
            "--config", str(cfg),
            "--warm-start", str(CKPT)
        ]

        print("==========================================", flush=True)
        print(f"Starting job {idx+1}/{len(JOBS)}: {name}", flush=True)
        print("Command:", " ".join(cmd), flush=True)
        print(f"Log: {log_file}", flush=True)
        print("==========================================", flush=True)

        with open(log_file, "a", encoding="utf-8") as out:
            proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                env=dict(os.environ, PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1"),
                stdout=out,
                stderr=subprocess.STDOUT
            )
            write_status("running", current_job=name, index=idx+1, total=len(JOBS), pid=proc.pid)
            exit_code = proc.wait()

        if exit_code != 0:
            write_status("failed", current_job=name, index=idx+1, exit_code=exit_code)
            print(f"Job {name} failed with exit code {exit_code}! Aborting queue.", flush=True)
            sys.exit(exit_code)

        write_status("job_completed", current_job=name, index=idx+1, total=len(JOBS))
        print(f"Job {name} completed successfully!", flush=True)

    write_status("all_completed", total_jobs=len(JOBS))
    print("All 3 DiC ablation jobs completed successfully!", flush=True)

if __name__ == "__main__":
    main()
