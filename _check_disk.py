import os, glob, json
# Check disk usage
os.chdir("/root/Workspace/xy/DiT")
st = os.statvfs("/root")
total_gb = st.f_blocks * st.f_frsize / 1e9
free_gb = st.f_bavail * st.f_frsize / 1e9
used_gb = total_gb - free_gb
print(f"Disk: {used_gb:.0f}G used / {total_gb:.0f}G total / {free_gb:.0f}G free")

# Estimate per-ckpt size from s14 test run
ckpt_dirs = glob.glob("5script/results/s14_ws_ddpm/*/checkpoints/")
if ckpt_dirs:
    d = ckpt_dirs[0]
    pts = glob.glob(d + "*.pt")
    if pts:
        sz = os.path.getsize(pts[0]) / 1e6
        print(f"s14 (WS/2) ckpt size: {sz:.0f} MB each")
        # With ckpt_keep=60: 60 * sz = max disk per run
        print(f"  max per run (60 kept): {60*sz/1e3:.1f} GB")

ckpt_dirs2 = glob.glob("5script/results/s16_s_ddpm/*/checkpoints/")
if ckpt_dirs2:
    d = ckpt_dirs2[0]
    pts = glob.glob(d + "*.pt")
    if pts:
        sz = os.path.getsize(pts[0]) / 1e6
        print(f"s16 (S/2) ckpt size: {sz:.0f} MB each")
        print(f"  max per run (60 kept): {60*sz/1e3:.1f} GB")

# Current results dir sizes
import subprocess
for series in ["s14_ws_ddpm", "s15_ws_flow", "s16_s_ddpm", "s17_s_flow"]:
    path = f"5script/results/{series}"
    if os.path.exists(path):
        total = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total += os.path.getsize(fp)
        print(f"  {series}: {total/1e6:.0f} MB")
    else:
        print(f"  {series}: (not yet created)")

# Eval samples also take space
for series in ["s14_ws_ddpm", "s15_ws_flow", "s16_s_ddpm", "s17_s_flow"]:
    path = f"5script/results/{series}"
    if os.path.exists(path):
        eval_dirs = glob.glob(f"{path}/*/checkpoints/eval_samples/*/")
        eval_sz = 0
        for ed in eval_dirs:
            for dirpath, dirnames, filenames in os.walk(ed):
                for f in filenames:
                    eval_sz += os.path.getsize(os.path.join(dirpath, f))
        if eval_sz > 0:
            print(f"  {series} eval_samples: {eval_sz/1e6:.0f} MB")
