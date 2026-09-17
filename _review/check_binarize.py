"""我们的数据是二值还是灰度? 管线里有没有二值化? HCSU wild 是二值还是灰度?
决定"要不要二值化"。"""
import os, csv, collections
import numpy as np
from PIL import Image

OUR = "/root/Workspace/xy/DiT"
our_csv = f"{OUR}/assets/train_fame-kxl-tj-px60.csv"
rows = list(csv.DictReader(open(our_csv, encoding="utf-8")))


def stats(paths, tag, n=150):
    lv, dark_frac, means = [], [], []
    for p in paths[:n]:
        if not os.path.exists(p):
            continue
        a = np.asarray(Image.open(p).convert("L"))
        u = np.unique(a)
        lv.append(len(u))
        dark_frac.append(float((a < 128).mean()))
        means.append(float(a.mean()))
    if not lv:
        print(f"{tag}: 无可用样本"); return
    lv = np.array(lv)
    print(f"{tag}  n={len(lv)}")
    print(f"   灰阶数  min={lv.min()} med={int(np.median(lv))} max={lv.max()}"
          f"   **二值(<=2 阶)占比 {100*(lv<=2).mean():.1f}%**")
    print(f"   暗像素占比 med={np.median(dark_frac):.3f}   均值 med={np.median(means):.1f}")


print("=== 我们的 image_path (原字) ===")
stats([os.path.join(OUR, r["image_path"]) for r in rows], "ours/img")
print("\n=== 我们的 std_path (标准字形骨架) ===")
stats([os.path.join(OUR, r["std_path"]) for r in rows if r["std_path"]], "ours/std")

# HCSU wild 抽样
ROOT = "/root/Workspace/xy/HCSU/wild_extract"
import random
random.seed(3)
wf = []
for d in sorted(os.listdir(ROOT)):
    dp = os.path.join(ROOT, d)
    if os.path.isdir(dp):
        for f in os.listdir(dp):
            if f.lower().endswith(".png"):
                wf.append(os.path.join(dp, f))
random.shuffle(wf)
print("\n=== HCSU wild (原图) ===")
stats(wf, "hcsu/wild")

# 管线里有没有二值化
print("\n=== 管线里搜 '二值/binari/threshold' ===")
import subprocess
for pat in ["binar", "threshold", "point(lambda", "convert\\(\"1\"\\)", "> 128", ">128"]:
    r = subprocess.run(["grep", "-rl", pat, f"{OUR}/src", f"{OUR}/tools", "--include=*.py"],
                       capture_output=True, text=True)
    fs = [x for x in r.stdout.split() if x]
    print(f"   {pat!r}: {len(fs)} 个文件  {fs[:4]}")
