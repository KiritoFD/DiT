"""检查 poster 用到的 png 到底有没有内容 (墨占比 + 均值)。"""
import glob

import numpy as np
from PIL import Image

for pat in ("g0.png", "g1.png", "gt0.png"):
    fs = sorted(glob.glob(f"assets/results/_calib_v34e2e_big/**/{pat}",
                          recursive=True))[:6]
    for f in fs:
        a = np.asarray(Image.open(f).convert("L"), np.float32) / 255.0
        print(f"{pat} ink={(a < 0.5).mean():.3f} mean={a.mean():.3f}  {f}")

fs = sorted(glob.glob("assets/results/_calib_v34e2e_big/eval_samples_ctrl/**/g0.png",
                      recursive=True))[:3]
for f in fs:
    a = np.asarray(Image.open(f).convert("L"), np.float32) / 255.0
    print(f"input_g ink={(a < 0.5).mean():.3f} mean={a.mean():.3f}  {f}")
