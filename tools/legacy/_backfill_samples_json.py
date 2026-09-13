"""Backfill samples.json for all step dirs missing it."""
import os, sys, glob, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
base = "/root/Workspace/xy/DiT/5script/results/s10_b4_grey_clear"
count = 0
for subdir in ["eval_samples", "show_samples", "seen_samples"]:
    for run in sorted(os.listdir(base)):
        if not run.startswith("20"): continue
        d = os.path.join(base, run, "checkpoints", subdir)
        if not os.path.isdir(d): continue
        for step_dir in sorted(os.listdir(d)):
            full = os.path.join(d, step_dir)
            if not os.path.isdir(full): continue
            sj = os.path.join(full, "samples.json")
            if os.path.exists(sj): continue
            n = len(glob.glob(os.path.join(full, "sample*.png")))
            step = int(step_dir.replace("step", ""))
            with open(sj, "w") as f:
                json.dump({"step": step, "n": n, "cfg": 4.0, "ddim_steps": 50}, f)
            count += 1
            print(f"wrote {subdir}/{step_dir} (n={n})")
print(f"\nTotal backfilled: {count}")
