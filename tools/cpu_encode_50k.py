# -*- coding: utf-8 -*-
"""50k 数据集的**多进程 CPU** VAE 编码（GPU 正被训练占用时用）。

## 为什么不用 GPU 版
训练（xattn）已占 21.6/24.5 GiB，只剩 2.5 GiB -> 跑 GPU encode 会 OOM 掉训练。
而本机有 **128 核**、负载才 ~15，CPU 编码可以完全并行、不挤训练。

## 做法
把 CSV 切成 `--nproc` 份，每份**独立调用 `src.data.vae_io.encode_csv`**
（device=cpu）写进自己的临时子目录，最后统一改名为
`shard_{proc:02d}_{k:05d}.npz` 并合并到 `--out`。

⚠ 复用 `encode_csv` 而不是自己写 VAE 前向 —— 保证 transform / dtype /
shard 格式与 GPU 版**逐位一致**。

## 断点安全
续跑时扫描 `--out` 里已有的 shard，收集已完成的 img_id，
每个进程只编码自己那份里**还没做过**的行。

## 用法
    python tools/cpu_encode_50k.py --csv assets/train_50k.csv --col image_path \\
        --out data/50k/shards_img --nproc 32 --threads 4 --batch 8
    python tools/cpu_encode_50k.py --csv assets/train_50k.csv --col std_path \\
        --out data/50k/shards_std --nproc 32 --threads 4 --batch 8
"""
import argparse
import csv
import glob
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

VAE = "data/pretrained/pretrained_models/sd-vae-ft-ema"


def done_ids(out_dir):
    """扫已有 shard，返回已编码的 img_id 集合（断点续跑用）。"""
    import numpy as np
    s = set()
    for sp in glob.glob(os.path.join(out_dir, "shard_*.npz")):
        try:
            with np.load(sp) as d:
                s.update(int(x) for x in d["img_ids"])
        except Exception:
            print(f"[warn] 读不了 {sp}，忽略")
    return s


def img_id_of(path):
    """从 `<数字>.png` 文件名取 id（与 latent_dataset.extract_img_id 同规则）。"""
    import re
    m = re.search(r"(\d+)\.png$", path)
    if not m:
        raise ValueError(f"无法从 {path!r} 提取 img_id")
    return int(m.group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--col", required=True, help="用哪一列当输入路径 (image_path / std_path)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--nproc", type=int, default=32)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--transform", default="gray")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
    have = done_ids(a.out)
    todo = [r for r in rows if img_id_of(r[a.col]) not in have]
    print(f"[cpu-encode] csv={a.csv} 共 {len(rows)} 行, 已完成 {len(have)}, "
          f"待编码 {len(todo)}")
    if not todo:
        print("[cpu-encode] 全部完成，无需编码")
        return

    # 切成 nproc 份
    chunks = [[] for _ in range(a.nproc)]
    for i, r in enumerate(todo):
        chunks[i % a.nproc].append(r)

    procs = []
    for pi, ch in enumerate(chunks):
        if not ch:
            continue
        tmp_csv = f"/tmp/_enc50k_{os.path.basename(a.out)}_{pi:02d}.csv"
        tmp_dir = os.path.join(a.out, f"_p{pi:02d}")
        os.makedirs(tmp_dir, exist_ok=True)
        with open(tmp_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image_path", "character", "script", "calligrapher"])
            for r in ch:
                w.writerow([r[a.col], r.get("character", ""),
                            r.get("script", ""), r.get("calligrapher", "")])
        # nice 降级: 不挤正在跑的训练
        cmd = (f"nice -n 10 {sys.executable} -m src.data.vae_io "
               f"--csv {tmp_csv} --out {tmp_dir} --transform {a.transform} "
               f"--vae-path {VAE} --batch {a.batch} --shard-size 5000 "
               f"--workers 0 --device cpu")
        env = dict(os.environ, OMP_NUM_THREADS=str(a.threads),
                   MKL_NUM_THREADS=str(a.threads))
        # ⚠ 捕获每个子进程的输出到独立日志 —— 上一版丢了 DEVNULL，导致
        #   48 个进程只剩 9 个活着时**完全看不到原因**。
        logf = open(os.path.join(a.out, f"_log_p{pi:02d}.txt"), "w")
        procs.append((pi, tmp_dir, subprocess.Popen(cmd, shell=True, env=env,
                                                    stdout=logf,
                                                    stderr=subprocess.STDOUT),
                      logf))
    print(f"[cpu-encode] 启动 {len(procs)} 个进程 (每个 {a.threads} 线程, nice 10)")

    for pi, tmp_dir, p, logf in procs:
        rc = p.wait()
        logf.close()
        print(f"  proc {pi:02d} 结束 rc={rc}", flush=True)

    # 合并: 改名到 out/，删临时目录
    n = 0
    for pi, tmp_dir, _, _ in procs:
        for k, sp in enumerate(sorted(glob.glob(os.path.join(tmp_dir, "shard_*.npz")))):
            dst = os.path.join(a.out, f"shard_{pi:02d}_{k:05d}.npz")
            shutil.move(sp, dst)
            n += 1
        shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"[cpu-encode] 合并 {n} 个 shard -> {a.out}")

    import numpy as np
    tot = 0
    for sp in glob.glob(os.path.join(a.out, "shard_*.npz")):
        with np.load(sp) as d:
            tot += d["img_ids"].shape[0]
    print(f"[cpu-encode] DONE {a.out}: {tot} latents")


if __name__ == "__main__":
    main()
