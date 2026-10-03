"""列出所有实验的最新 ckpt 及其 callig_id_map，判断能否用新 eval 集评。"""
import glob
import os
import sys

import torch

os.chdir("/root/Workspace/xy/DiT")

runs = sorted(os.path.basename(p.rstrip("/"))
              for p in glob.glob("assets/results/*") if os.path.isdir(p))
print(f"  共 {len(runs)} 个实验\n")
print(f"  {'实验':<52}{'callig_id_map':<40}{'书家数':>6}")
print("  " + "-" * 98)
ok = []
for run in runs:
    cks = sorted(glob.glob(f"assets/results/{run}/*/checkpoints/[0-9]*.pt"),
                 key=lambda p: int(os.path.basename(p)[:-3]))
    if not cks:
        continue
    ck = cks[-1]
    try:
        d = torch.load(ck, map_location="cpu", weights_only=False)
        a = d.get("args")
        a = vars(a) if hasattr(a, "__dict__") else dict(a or {})
        m = a.get("callig_id_map") or "None"
        n = a.get("num_calligraphers", "?")
    except Exception as e:
        m, n = f"ERR:{type(e).__name__}", "?"
    print(f"  {run:<52}{str(m)[:38]:<40}{str(n):>6}")
    if "50k" in str(m):
        ok.append(run)

print(f"\n  === 能用新 eval 集评的（callig_id_map 含 50k）: {len(ok)} 个 ===")
for r in ok:
    print(f"    {r}")
