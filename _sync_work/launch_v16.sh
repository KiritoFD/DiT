#!/bin/bash
# v16 串行长训：8 个主题一个接一个（用户指定：串行 + lr 1e-5 + 系列名 v16）。
# 判读以 Diff 走平为准。
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT

for t in 沈周-行 伊秉绶-行 傅山-行 伊秉绶-隶 徐渭-行 张即之-楷 郑道昭-楷 张裕钊-楷; do
  bash _sync_work/run_v16.sh "$t"
done
/opt/conda/envs/cu121/bin/python - <<'PY'
import glob, os, re
RE_D = re.compile(r"\(step=(\d+)\).*?Diff: ([0-9.]+)")
RE_E = re.compile(r"step=(\d+) set=fewshot n=(\d+) ssim=([\d.]+)")
print("\n==== v16 汇总 ====")
for lg in sorted(glob.glob("logs/v16_series/*.log")):
    ds, es = [], []
    for line in open(lg, encoding="utf-8", errors="replace"):
        m = RE_D.search(line)
        if m: ds.append(float(m.group(2)))
        m = RE_E.search(line)
        if m: es.append((int(m.group(1)), float(m.group(3))))
    if not ds: continue
    name = os.path.basename(lg).split("_0921-")[0]
    tail = ds[-500:]
    print(f"{name:<22} Diff 首={ds[0]:.3f} 末={ds[-1]:.3f} 末500均值={sum(tail)/len(tail):.3f}"
          f" | ssim 首={es[0][1]:.4f} 峰={max(e[1] for e in es):.4f}" if es else
          f"{name:<22} Diff 首={ds[0]:.3f} 末={ds[-1]:.3f}")
PY
echo "[v16] ALL DONE $(date '+%m-%d %H:%M')"
