"""Check remote eval_samples/show_samples/seen_samples dirs and samples.json status."""
import os, sys, glob, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
base = "/root/Workspace/xy/DiT/5script/results/s10_b4_grey_clear"
for subdir in ["eval_samples", "show_samples", "seen_samples"]:
    print(f"\n=== {subdir} ===")
    for run in sorted(os.listdir(base)):
        if not run.startswith("20"): continue
        d = os.path.join(base, run, "checkpoints", subdir)
        if not os.path.isdir(d): continue
        for step_dir in sorted(os.listdir(d)):
            full = os.path.join(d, step_dir)
            if not os.path.isdir(full): continue
            has_json = os.path.exists(os.path.join(full, "samples.json"))
            n = len(glob.glob(os.path.join(full, "sample*.png")))
            print(f"  {run[:25]}/{step_dir}: n={n}, json={has_json}")
