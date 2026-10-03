"""把某个 ckpt 目录下所有 eval_auto_*.json 打成一张表 (按 step 排序)。"""
import glob
import json
import os
import sys

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)

d = sys.argv[1] if len(sys.argv) > 1 else \
    "exp-std/runs/20261003-033702-v46-std-adaln4-top10-p0.2/checkpoints"
keys = ["step", "ssim", "lpips", "skel_iou", "ink_ssim", "ink_iou", "frag",
        "hole", "hole_gt", "n", "nn_ssim", "cal_enrich"]


def _step(p):
    return int("".join(c for c in os.path.basename(p) if c.isdigit()) or 0)


rows = [json.load(open(f, encoding="utf-8"))
        for f in sorted(glob.glob(os.path.join(d, "eval_auto_*.json")), key=_step)]
print(f"[{d}]  {len(rows)} 个判分点")
print("  ".join(f"{k:>9}" for k in keys))
for r in rows:
    out = []
    for k in keys:
        v = r.get(k)
        out.append(f"{v:>9.4f}" if isinstance(v, float) else f"{str(v):>9}")
    print("  ".join(out))
