"""重画某次 in_mem_eval 的 poster (不训练/不推理, 只读已落盘样本 + 摘要)。

用法:
  POSTER_CELL=256 POSTER_NMAX=6 python tools/repost_poster.py \
      --results-dir exp-std/runs --set eval200 \
      --train-csv exp-std/csv/train.csv --eval-csv exp-std/csv/eval200.csv --n-eval 200
环境变量 POSTER_CELL / POSTER_NMAX 控制每格像素与最大列数 (原生 256 才看得清笔画)。
"""
import argparse
import os
import sys

ROOT = "/root/Workspace/xy/DiT"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--results-dir", required=True)
ap.add_argument("--set", dest="set_name", required=True)
ap.add_argument("--train-csv", default=None)
ap.add_argument("--eval-csv", default=None)
ap.add_argument("--n-eval", type=int, default=0)
a = ap.parse_args()

from src.eval.in_mem_eval import render_poster  # noqa: E402

p = render_poster(a.results_dir, a.set_name, train_csv=a.train_csv,
                  eval_csv=a.eval_csv, n_eval=a.n_eval)
print(f"[repost] poster = {p}")
