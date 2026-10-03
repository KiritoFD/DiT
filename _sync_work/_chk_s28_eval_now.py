# -*- coding: utf-8 -*-
import os, sys, glob, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir("/root/Workspace/xy/DiT")

L = sorted(glob.glob("assets/results/s28_std_dino_pretrain/2026*/"))[-1]
print(f"run: {L}")
print("=== s28 (clean+PCA+OT) eval ===")
for f in sorted(glob.glob(f"{L}checkpoints/eval_auto_*.json")):
    d = json.load(open(f))
    print(f"  step {d['step']:>5}  ssim={d['ssim']:.4f}  lpips={d['lpips']:.4f}  mse={d['mse']:.4f}")

# 最新 step
import subprocess
log = f"{L}log.txt"
step = subprocess.run(["grep","-a","-o","step=[0-9]*",log],capture_output=True,text=True).stdout.strip().split("\n")
if step: print(f"\nlatest step: {step[-1]}")
loss = subprocess.run(["grep","-a","-o","Total: [0-9.]*",log],capture_output=True,text=True).stdout.strip().split("\n")
if loss: print(f"latest Total: {loss[-1]}")

print("\n=== s28→s29 监控状态 ===")
r = subprocess.run(["tmux","ls"],capture_output=True,text=True).stdout
for line in r.split("\n"):
    if "s28" in line or "s29" in line:
        print(f"  {line}")
wlog = "_sync_work/s28_to_s29_watch.log"
if os.path.isfile(wlog):
    print("watch log:", open(wlog).read()[:200])
