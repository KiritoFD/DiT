import os, json, glob
sys_path = "/root/Workspace/xy/DiT/5script/results/s10_b4_grey_clear"
for ckpt_dir in sorted(glob.glob(os.path.join(sys_path, "*/checkpoints"))):
    for step_dir in sorted(glob.glob(os.path.join(ckpt_dir, "eval_samples/step*"))):
        sj = os.path.join(step_dir, "samples.json")
        if os.path.exists(sj):
            continue
        step = int(os.path.basename(step_dir).replace("step", ""))
        n = len(glob.glob(os.path.join(step_dir, "sample*.png")))
        with open(sj, "w") as f:
            json.dump({"step": step, "n": n, "cfg": 4.0, "ddim_steps": 50}, f)
        print(f"wrote {sj} (n={n})")
    for step_dir in sorted(glob.glob(os.path.join(ckpt_dir, "show_samples/step*"))):
        sj = os.path.join(step_dir, "samples.json")
        if os.path.exists(sj):
            continue
        step = int(os.path.basename(step_dir).replace("step", ""))
        n = len(glob.glob(os.path.join(step_dir, "sample*.png")))
        with open(sj, "w") as f:
            json.dump({"step": step, "n": n, "cfg": 4.0, "ddim_steps": 50}, f)
        print(f"wrote {sj} (n={n})")
    for step_dir in sorted(glob.glob(os.path.join(ckpt_dir, "seen_samples/step*"))):
        sj = os.path.join(step_dir, "samples.json")
        if os.path.exists(sj):
            continue
        step = int(os.path.basename(step_dir).replace("step", ""))
        n = len(glob.glob(os.path.join(step_dir, "sample*.png")))
        with open(sj, "w") as f:
            json.dump({"step": step, "n": n, "cfg": 4.0, "ddim_steps": 50}, f)
        print(f"wrote {sj} (n={n})")
