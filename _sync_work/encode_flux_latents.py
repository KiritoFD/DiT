"""用 FLUX AE(16ch) 把**整份数据**编码成 shards, 分片格式与既有约定完全一致。

输出: <out-dir>/shard_%05d.npz   {'latents': (n,16,32,32) fp16, 'img_ids': (n,) int64}
      <out-dir>/manifest.json
用法:
  # 真迹图 (信号实验 / 换 VAE 的底座)
  python tools/encode_flux_latents.py --img-dir data/top10_style23/imgs \
      --out-dir exp-std/data/shards_img_flux16 --batch 256 --workers 16
  # std 条件图 (若换 VAE 部署侧也要)
  python tools/encode_flux_latents.py --img-dir data/top10_style23/std \
      --out-dir exp-std/data/shards_std_flux16 --batch 256 --workers 16
提速: 大 batch + 多 worker 预取 + bf16 + cudnn.benchmark (目标: 吃满卡, 图/秒 ~数百)
"""
import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np
import torch as th
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from flux_ae import encode_flux, load_flux_ae  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--img-dir", required=True)
ap.add_argument("--out-dir", required=True)
# ★ batch 不能开大: Flux AE 第一个 encoder block 在 256x256 全分辨率 / 128ch 上跑
#   group_norm, batch512 时单层激活 512x512x256x256 -> 要 16 GiB -> OOM(实测踩过)。
#   64 时约需 2 GiB, 稳。
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--workers", type=int, default=16)
ap.add_argument("--shard-size", type=int, default=5120)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--size", type=int, default=256)
ap.add_argument("--skip-existing", action="store_true")
ap.add_argument("--legacy-ckpt", default="")
a = ap.parse_args()

os.makedirs(a.out_dir, exist_ok=True)
dev = th.device("cuda" if th.cuda.is_available() else "cpu")
th.backends.cudnn.benchmark = False     # autotune 会申请巨大 workspace(实测尖峰 22.7G)


def enc_any(vae, x, scale, shift, min_b=4):
    """编码一个 batch; 若 OOM 就**折半递归**, 保证不崩(显存波动/并发时保命)。"""
    try:
        return encode_flux(vae, x.to(dev, non_blocking=True), scale, shift)
    except th.cuda.OutOfMemoryError:
        th.cuda.empty_cache()
        if x.shape[0] <= min_b:
            raise
        h = x.shape[0] // 2
        print(f"    [oom] batch {x.shape[0]} 折半 -> {h}", flush=True)
        return th.cat([enc_any(vae, x[:h], scale, shift, min_b),
                       enc_any(vae, x[h:], scale, shift, min_b)], 0)

files = sorted(glob.glob(os.path.join(a.img_dir, "*.png")))
files = [f for f in files if re.fullmatch(r"\d+", os.path.basename(f)[:-4] or "")]
if a.limit:
    files = files[:a.limit]
print(f"[in] {a.img_dir}: {len(files)} 张 PNG")
assert files, "没找到数字命名的 PNG"


class Imgs(Dataset):
    def __init__(self, fs, size):
        self.fs, self.size = fs, size

    def __len__(self):
        return len(self.fs)

    def __getitem__(self, i):
        p = self.fs[i]
        im = Image.open(p).convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        x = np.asarray(im, np.float32) / 255.0
        return int(os.path.basename(p)[:-4]), th.from_numpy(x).permute(2, 0, 1)


def collate(bs):
    ids = [b[0] for b in bs]
    x = th.stack([b[1] for b in bs])
    return ids, x


vae, SCALE, SHIFT = load_flux_ae(dev, legacy_ckpt=a.legacy_ckpt)
print(f"[ae] load 完成, scale={SCALE} shift={SHIFT}")

dl = DataLoader(Imgs(files, a.size), batch_size=a.batch, num_workers=a.workers,
                shuffle=False, drop_last=False, pin_memory=True, collate_fn=collate,
                prefetch_factor=6, persistent_workers=True)

buf_z, buf_id, shard_i, done, t0 = [], [], 0, 0, time.time()
man = []


def flush():
    global buf_z, buf_id, shard_i
    if not buf_z:
        return
    z = th.cat(buf_z, 0).numpy().astype(np.float16)
    ids = np.array(buf_id, dtype=np.int64)
    p = os.path.join(a.out_dir, f"shard_{shard_i:05d}.npz")
    np.savez_compressed(p, latents=z, img_ids=ids)
    man.append({"shard": os.path.basename(p), "n": int(z.shape[0]), "path": p})
    print(f"  [shard] {p}  n={z.shape[0]}  {z.shape}")
    buf_z, buf_id, shard_i = [], [], shard_i + 1


with th.no_grad():
    for bi, (ids, x) in enumerate(dl):
        zz = enc_any(vae, x, SCALE, SHIFT)
        buf_z.append(zz.to(th.float16).cpu())
        buf_id += ids
        done += len(ids)
        if len(buf_id) >= a.shard_size:
            flush()
        del x, zz
        if bi % 1 == 0 and done < len(files):
            th.cuda.empty_cache()
        if bi % 10 == 0 or done >= len(files):
            el = time.time() - t0
            print(f"  {done}/{len(files)}  {done / max(el, 1e-9):.0f} img/s  "
                  f"已用 {el / 60:.1f} min  ETA {(len(files) - done) / max(done / el, 1e-9) / 60:.1f} min",
                  flush=True)
flush()
with open(os.path.join(a.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
    json.dump({"img_dir": a.img_dir, "size": a.size, "scale": SCALE, "shift": SHIFT,
               "total": int(sum(m["n"] for m in man)), "shards": man}, f, indent=2)
print(f"[done] {done} 张 -> {a.out_dir}  用时 {(time.time() - t0) / 60:.1f} min")
