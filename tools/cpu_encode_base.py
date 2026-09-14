# -*- coding: utf-8 -*-
"""
cpu_encode_base.py — fame-tj-uc base 多进程 CPU encode (v9 模式, shard 即时落盘).

任务 (256x256 -> VAE latent (4,32,32) fp16):
  img       — base csv 54,892 行 GT 图
  aux_skel3 — data/skel/final_skel3_base/{img_id}.png
  aux_canny — data/aux/final_canny_base/{img_id}.png
  std_skel  — data/skel/std_skel3_base_png/{uid}.png (5,627 唯一 (script,char), uid 8600000+)

并行: nproc 进程 x ithr 线程 (oneDNN), nice 降级不挤训练.
断点安全: 每 worker 每满 2,592 张即时落盘一个 shard_{me}_{k}.npz; 尾部也落盘;
续跑时扫已有 shard 的 img_ids 跳过已完成.
"""
import argparse
import csv
import glob
import multiprocessing as mp
import os
import re
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir("/root/Workspace/xy/DiT")

VAE_PATH = "data/pretrained/pretrained_models/sd-vae-ft-ema"
SF = 0.18215
SHARD = 512

PHASES = {
    "img": "data/latents/final_latents_base_shards",
    "aux_skel3": "data/skel/aux_skel3_latents_base",
    "aux_canny": "data/aux/aux_canny_latents_base",
    "std_skel": "data/skel/std_skel3_latents_base",
}


def base_rows():
    rows = []
    for r in csv.DictReader(open("assets/train_base_noaug.csv", encoding="utf-8")):
        iid = int(re.search(r"(\d+)\.png", r["image_path"]).group(1))
        rows.append((iid, r["image_path"]))
    return rows


def load_tasks(phase, rows):
    if phase == "img":
        return rows
    if phase == "aux_skel3":
        return [(iid, f"data/skel/final_skel3_base/{iid}.png") for iid, _ in rows]
    if phase == "aux_canny":
        return [(iid, f"data/aux/final_canny_base/{iid}.png") for iid, _ in rows]
    if phase == "std_skel":
        return [(int(os.path.basename(p)[:-4]), p)
                for p in sorted(glob.glob("data/skel/std_skel3_base_png/*.png"))]
    raise ValueError(phase)


def worker(me, nproc, ithr, batch, tasks, out_dir, tag, q):
    os.environ["OMP_NUM_THREADS"] = str(ithr)
    os.environ["MKL_NUM_THREADS"] = str(ithr)
    try:
        os.nice(10)
    except Exception:
        pass
    import torch
    torch.set_num_threads(ithr)
    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(VAE_PATH).eval()

    def enc(x):
        with torch.no_grad():
            return (vae.encode(x).latent_dist.mode() * SF).to(torch.float16).cpu().numpy()

    mine = tasks[me::nproc]
    done = set()
    for sp in glob.glob(os.path.join(out_dir, "shard_*.npz")):
        try:
            with np.load(sp) as d:
                done.update(int(i) for i in d["img_ids"])
        except Exception:
            pass
    todo = [(i, p) for i, p in mine if i not in done]
    print(f"[p{me} {tag}] todo {len(todo)}/{len(mine)} (already {len(done)})", flush=True)
    if not todo:
        q.put((me, 0))
        return

    t0 = time.time()
    acc = []          # 待编码 batch
    acc_ids = []      # 与 acc 对应
    enc_ids = []      # 已编码待落盘
    enc_lat = []
    n_shard = 0
    n_fail = 0
    n_ok = 0

    def flush(force=False):
        nonlocal enc_ids, enc_lat, n_shard
        while force or len(enc_ids) >= SHARD:
            if not enc_ids:
                return
            m = min(SHARD, len(enc_ids)) if not force else len(enc_ids)
            np.savez(os.path.join(out_dir, f"shard_{me:02d}_{n_shard:04d}.npz"),
                     latents=np.stack(enc_lat[:m]).astype(np.float16),
                     img_ids=np.array(enc_ids[:m], dtype=np.int64))
            n_shard += 1
            enc_ids = enc_ids[m:]
            enc_lat = enc_lat[m:]
            if not enc_ids:
                break

    for k, (iid, path) in enumerate(todo):
        try:
            if args_tag == "img":
                a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0 * 2 - 1
                acc.append(a.transpose(2, 0, 1))
            else:
                a = np.asarray(Image.open(path).convert("L"))
                acc.append((a.astype(np.float32) / 255.0 * 2 - 1)[None])
            acc_ids.append(iid)
            if len(acc) >= batch:
                x = np.stack(acc)
                if x.shape[1] == 1:
                    x = np.repeat(x, 3, axis=1)
                z = enc(torch.from_numpy(x))
                enc_ids.extend(acc_ids)
                enc_lat.extend(z)
                acc.clear()
                acc_ids.clear()
                flush()
        except Exception as e:
            n_fail += 1
            print(f"[p{me} {tag}] FAIL {path}: {e}", flush=True)
        if (k + 1) % 500 == 0:
            print(f"[p{me} {tag}] {k+1}/{len(todo)} ({(k+1)/(time.time()-t0):.1f}/s)", flush=True)
    if acc:
        x = np.stack(acc)
        if x.shape[1] == 1:
            x = np.repeat(x, 3, axis=1)
        z = enc(torch.from_numpy(x))
        enc_ids.extend(acc_ids)
        enc_lat.extend(z)
        acc.clear()
        acc_ids.clear()
    flush(force=True)
    dt = time.time() - t0
    print(f"[p{me} {tag}] DONE ok={len(enc_ids)+n_fail-n_fail} shards={n_shard} "
          f"fail={n_fail} in {dt:.0f}s ({len(todo)/max(dt,1e-6):.1f}/s)", flush=True)
    q.put((me, len(todo) - n_fail))


args_tag = None  # worker 内通过 mp 传参方式拿 phase tag (见 run())


def run(args):
    global args_tag
    args_tag = args.phase
    out_dir = PHASES[args.phase]
    os.makedirs(out_dir, exist_ok=True)
    rows = base_rows() if args.phase == "img" else []
    tasks = load_tasks(args.phase, rows)
    print(f"[{args.phase}] tasks: {len(tasks)} -> {out_dir}", flush=True)
    q = mp.Queue()
    procs = []
    for me in range(args.nproc):
        p = mp.Process(target=worker,
                       args=(me, args.nproc, args.threads, args.batch,
                             tasks, out_dir, args.phase, q))
        p.start()
        procs.append(p)
    n = 0
    for _ in procs:
        n += q.get()
    for p in procs:
        p.join()
    print(f"[{args.phase}] ALL DONE: {n} encodes", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["img", "aux_skel3", "aux_canny", "std_skel"])
    ap.add_argument("--nproc", type=int, default=16)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    run(args)
