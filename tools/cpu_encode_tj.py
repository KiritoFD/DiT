# -*- coding: utf-8 -*-
"""cpu_encode_tj.py — fame-tj-kxl tongji assets 多进程 CPU encode (v9 模式, 分阶段幂等).

量: img 8,976 + aux_skel3 8,976 + aux_canny 8,976 + std_skel 8,555 ≈ 35.5k 张 256x256.
并行: nproc 进程 x ithr 线程 (oneDNN SIMD), nice 10 降优先级不挤训练.
输出 shard npz (latents fp16 + img_ids), 与 latent_dataset 查找格式一致:
  data/latents/final_latents_tj_shards/    (image, transform gray)
  data/skel/aux_skel3_latents_tj/          (GT 图骨架化 + dil3, transform skel)
  data/aux/aux_canny_latents_tj/           (GT 图 canny 边缘, transform canny)
  data/skel/std_skel3_latents_tj/          (字体渲染标准字形, per (script,char) 唯一)

用法: python tools/cpu_encode_tj.py --phase img --nproc 16 --threads 4
      ... (每阶段独立; 产物已存在则跳过)
"""
import csv
import glob
import io
import multiprocessing as mp
import os
import sys
import time
import argparse

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = "/root/Workspace/xy/DiT"
os.chdir(BASE)
VAE_PATH = "data/pretrained/pretrained_models/sd-vae-ft-ema"
SF = 0.18215

PHASES = {
    "img": ("data/latents/final_latents_tj_shards", "gray"),
    "aux_skel3": ("data/skel/aux_skel3_latents_tj", "skel"),
    "aux_canny": ("data/aux/aux_canny_latents_tj", "canny"),
}
STD_OUT = "data/skel/std_skel3_latents_tj"
STD_SHARDS_DONE = os.path.join(STD_OUT, "shard_00000.npz")


def tf_gray(im):
    """图像 (3,256,256) [-1,1]."""
    arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0 * 2 - 1
    return arr.transpose(2, 0, 1)


def tf_skel(im, dilate=1):
    """GT 图 -> 骨架 (黑线白底) -> dilate 1 (~3px) -> (1,256,256) [-1,1]."""
    from scipy.ndimage import binary_dilation, generate_binary_structure
    try:
        from skimage.morphology import skeletonize
    except ImportError:
        raise
    a = np.asarray(im.convert("L"))
    sk = skeletonize(a < 127)
    if dilate:
        sk = binary_dilation(sk, generate_binary_structure(2, 2), iterations=dilate)
    img = np.where(sk, 0, 255).astype("uint8")
    return (img.astype(np.float32) / 255.0 * 2 - 1)[None]


def tf_canny(im):
    """GT 图 -> canny 边缘 (黑线白底) -> (1,256,256) [-1,1]."""
    import cv2
    a = np.asarray(im.convert("L"))
    edges = cv2.Canny(a, 80, 180)
    return (edges.astype(np.float32) / 255.0 * 2 - 1)[None]


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

    out_f = os.path.join("/tmp/tj_enc", f"w{me}_{tag}.npz")
    os.makedirs("/tmp/tj_enc", exist_ok=True)
    if os.path.exists(out_f):
        print(f"[p{me} {tag}] skip (exists)", flush=True)
        q.put((me, 0))
        return

    mine = tasks[me::nproc]
    t0 = time.time()
    acc = []
    ids = []
    lat = []

    def flush():
        if not acc:
            return
        x = np.stack(acc)
        if x.shape[1] == 1:
            x = np.repeat(x, 3, axis=1)
        z = enc(torch.from_numpy(x))
        for j, iid in enumerate(ids_buf):
            ids.append(iid)
            lat.append(z[j])
        acc.clear()
        ids_buf.clear()

    ids_buf = []
    done = 0
    for iid, arr in mine:
        acc.append(arr)
        ids_buf.append(iid)
        if len(acc) >= batch:
            flush()
            done += batch
            if (done // batch) % 20 == 0:
                print(f"[p{me} {tag}] {done}/{len(mine)}", flush=True)
    flush()
    a_ids = np.array(ids, dtype=np.int64)
    lat_arr = np.stack(lat) if a_ids.size else np.zeros((0, 4, 32, 32), np.float16)
    np.savez(out_f, ids=a_ids, lat=lat_arr)
    dt = time.time() - t0
    print(f"[p{me} {tag}] DONE {len(ids)} in {dt:.0f}s ({len(ids)/max(dt,1e-6):.1f} img/s)",
          flush=True)
    q.put((me, len(ids)))


def run_phase(name, tasks, out_dir, nproc, ithr, batch):
    os.makedirs(out_dir, exist_ok=True)
    done_marker = os.path.join(out_dir, ".done")
    if os.path.exists(done_marker):
        print(f"[{name}] skip (done marker)", flush=True)
        return
    q = mp.Queue()
    procs = []
    for me in range(nproc):
        p = mp.Process(target=worker,
                       args=(me, nproc, ithr, batch, tasks, out_dir, name, q))
        p.start()
        procs.append(p)
    n = 0
    for _ in procs:
        n += q.get()
    for p in procs:
        p.join()
    # 聚合 worker npz -> shard_XXXXX.npz (latent_dataset 查找格式)
    w_files = sorted(glob.glob(f"/tmp/tj_enc/w*_{name}.npz"))
    shard, ids = [], []
    n_shard = 0
    shard_size = 2592

    def flush_shard():
        nonlocal shard, ids, n_shard
        if not shard:
            return
        np.savez(os.path.join(out_dir, f"shard_{n_shard:05d}.npz"),
                 latents=np.stack(shard).astype(np.float16),
                 img_ids=np.array(ids, dtype=np.int64))
        n_shard += 1
        shard, ids = [], []

    for wf in w_files:
        with np.load(wf) as d:
            for j, iid in enumerate(d["ids"]):
                shard.append(d["lat"][j])
                ids.append(int(iid))
                if len(shard) >= shard_size:
                    flush_shard()
    flush_shard()
    open(done_marker, "w").close()
    print(f"[{name}] {n} encodes -> {n_shard} shards in {out_dir}/", flush=True)


def std_skel_tasks():
    """渲染标准字形骨架 (per (script,char) 唯一), 任务 = (uid, (1,256,256))."""
    from scipy.ndimage import binary_dilation, generate_binary_structure
    from skimage.morphology import skeletonize
    from PIL import ImageFont, ImageDraw
    SCRIPT_FONT = {
        "楷": ["simkai.ttf", "STKAITI.TTF", "NotoSerifSC-VF.ttf"],
        "行": ["STXINGKA.TTF", "FZSTK.TTF"],
        "隶": ["SIMLI.TTF", "STLITI.TTF"],
    }
    font_cache = {}

    def render(ch, script):
        cands = list(SCRIPT_FONT.get(script, SCRIPT_FONT["楷"])) + ["simkai.ttf", "simhei.ttf"]
        seen = set()
        for f in cands:
            if f in seen:
                continue
            seen.add(f)
            fp = os.path.join("tools/fonts", f)
            if not os.path.isfile(fp):
                continue
            try:
                font = font_cache.setdefault(f, ImageFont.truetype(fp, 200))
            except Exception:
                continue
            img = Image.new("L", (256, 256), 255)
            ImageDraw.Draw(img).text((128, 128), ch, font=font, fill=0, anchor="mm")
            a = np.asarray(img)
            if (a < 250).sum() < 10:
                continue
            sk = skeletonize(a < 127)
            sk = binary_dilation(sk, generate_binary_structure(2, 2), iterations=1)
            return np.where(sk, 0, 255).astype("uint8").astype(np.float32) / 255.0 * 2 - 1
        return None

    rows = list(csv.DictReader(open("assets/train_fame_tj_kxl.csv", encoding="utf-8")))
    # tongji 行 img_id 段 950000+; std skel 按 (script,char) 唯一, uid = 8600000+order
    pairs = sorted({(r["script"], r["character"]) for r in rows
                    if "calli_tongji" in r["image_path"]})
    # 只对 fame 现有 std skel shard 覆盖不到的 (script,char) 渲染:
    covered = set()
    for sp in glob.glob("data/skel/std_skel3_latents_fame_sym/shard_*.npz"):
        with np.load(sp) as d:
            covered.update(int(i) for i in d["img_ids"])
    id2key = {}
    for r in rows:
        if "calli_tongji" in r["image_path"]:
            iid = int(re.search(r"(\d+)\.png", r["image_path"]).group(1))
            id2key[iid] = (r["script"], r["character"])
    tasks = []
    uid = 8600000
    key2uid = {}
    for script, ch in pairs:
        key2uid[(script, ch)] = uid
        uid += 1
    return pairs, key2uid, render, id2key


def run_std_skel(nproc, ithr, batch):
    """std skel: 渲染 + encode, shard 按 tongji img_id 展开 (uid -> img_id)."""
    import re as _re
    pairs, key2uid, render, id2key = std_skel_tasks()
    out_dir = STD_OUT
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(os.path.join(out_dir, ".done")):
        print("[std_skel] skip (done)", flush=True)
        return
    # 渲染 (CPU, 单进程足够快: 8555 张 PIL+skel ~1-2s each -> 数小时? 多进程)
    def render_worker(me, nproc, pairs, q):
        os.nice(10)
        tasks = []
        for k, (script, ch) in enumerate(pairs):
            if k % nproc != me:
                continue
            arr = render(ch, script)
            if arr is not None:
                tasks.append((key2uid[(script, ch)], arr[None]))
        q.put((me, tasks))

    q = mp.Queue()
    procs = []
    for me in range(nproc):
        p = mp.Process(target=render_worker, args=(me, nproc, pairs, q))
        p.start()
        procs.append(p)
    all_tasks = []
    for _ in procs:
        all_tasks.extend(q.get()[1])
    for p in procs:
        p.join()
    print(f"[std_skel] rendered {len(all_tasks)}/{len(pairs)}", flush=True)
    # encode + shard 按 img_id 展开
    q2 = mp.Queue()
    procs = []
    for me in range(nproc):
        p = mp.Process(target=worker, args=(me, nproc, ithr, batch, all_tasks,
                                            out_dir, "std_skel", q2))
        p.start()
        procs.append(p)
    for _ in procs:
        q2.get()
    for p in procs:
        p.join()
    # 聚合 + 展开: uid -> img_id (tongji 行)
    w_files = sorted(glob.glob(f"/tmp/tj_enc/w*_std_skel.npz"))
    uid2lat = {}
    for wf in w_files:
        with np.load(wf) as d:
            for j, uid in enumerate(d["ids"]):
                uid2lat[int(uid)] = d["lat"][j]
    shard, ids = [], []
    n_shard = 0

    def flush_shard():
        nonlocal shard, ids, n_shard
        if not shard:
            return
        np.savez(os.path.join(out_dir, f"shard_{n_shard:05d}.npz"),
                 latents=np.stack(shard).astype(np.float16),
                 img_ids=np.array(ids, dtype=np.int64))
        n_shard += 1
        shard, ids = [], []

    rows = list(csv.DictReader(open("assets/train_tongji_only.csv", encoding="utf-8")))
    n_have = 0
    for r in rows:
        iid = int(_re.search(r"(\d+)\.png", r["image_path"]).group(1))
        u = key2uid.get((r["script"], r["character"]))
        if u is None or u not in uid2lat:
            continue
        shard.append(uid2lat[u])
        ids.append(iid)
        if len(shard) >= 2592:
            flush_shard()
        n_have += 1
    flush_shard()
    open(os.path.join(out_dir, ".done"), "w").close()
    print(f"[std_skel] {n_have} rows covered -> {n_shard} shards", flush=True)


import re  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["img", "aux_skel3", "aux_canny", "std_skel"])
    ap.add_argument("--nproc", type=int, default=16)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    if args.phase == "std_skel":
        run_std_skel(args.nproc, args.threads, args.batch)
        return
    out_dir, transform = PHASES[args.phase]
    rows = list(csv.DictReader(open("assets/train_tongji_only.csv", encoding="utf-8")))
    tasks = []
    t0 = time.time()
    for r in rows:
        iid = int(re.search(r"(\d+)\.png", r["image_path"]).group(1))
        im = Image.open(r["image_path"])
        if transform == "gray":
            tasks.append((iid, tf_gray(im)))
        elif transform == "skel":
            tasks.append((iid, tf_skel(im)))
        else:
            tasks.append((iid, tf_canny(im)))
    print(f"[{args.phase}] precomputed {len(tasks)} in {time.time()-t0:.0f}s", flush=True)
    run_phase(args.phase, tasks, out_dir, args.nproc, args.threads, args.batch)


if __name__ == "__main__":
    main()
