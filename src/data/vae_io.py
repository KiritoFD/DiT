# -*- coding: utf-8 -*-
"""vae_io.py — 统一的高性能 VAE encode/decode 轮子 (encode 目标 latent shards).

设计要点
--------
- **并行读图**: torch DataLoader(num_workers) 取代旧 builder 的串行 PIL 循环
  （旧: 85k 图 ~12min 的瓶颈在单线程读图/灰度化，不是 VAE）。
- **GPU 高效**: channels_last + cudnn.benchmark + TF32；encode 默认 bf16（快且稳），
  decode 默认 fp32（sd-vae fp16 decode 会掉质量）。
- **可选编译**: `compile=True` 用 torch.compile（默认关；产物落 TORCHINDUCTOR_CACHE_DIR）。
- **统一 transform**: gray / skel(可 dilate) / canny 三种目标，覆盖
  image latent、实例骨架、canny、以及调用方自渲染的标准字形（传 tensor 直接 encode）。
- **API**: `VAEIO.encode(tensor)` / `VAEIO.decode(latents)`；`encode_csv()` 直接产 shards。

用法:
  # 构建图像 latent
  python -m src.data.vae_io --csv assets/train_fame3_sym_full.csv --out final_latents_fame_sym --transform gray
  # 构建 3px 骨架 aux
  python -m src.data.vae_io --csv assets/train_fame3_sym_full.csv --out aux_skel3_latents_sym --transform skel --skel-dilate 1
  # 基准
  python -m src.data.vae_io --bench --batch 64 --n 512
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time

import numpy as np
import torch

SCALING = 0.18215
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --------------------------------------------------------------------------- #
# transforms (CPU, DataLoader workers 里执行)
# --------------------------------------------------------------------------- #
def _to_img(arr, bg=255, line=0):
    """(H,W) 0..255 的线图 -> (3,H,W) [-1,1]; bg=底色, line=线色。"""
    a = np.stack([arr, arr, arr], 0).astype(np.float32) / 127.5 - 1.0
    return a


def _tf_gray(path, skel_dilate=0):
    from PIL import Image
    g = Image.open(path).convert("L")
    if g.size != (256, 256):
        g = g.resize((256, 256), Image.LANCZOS)
    return _to_img(np.asarray(g, np.float32))


def _tf_skel(path, skel_dilate=0):
    from PIL import Image
    from skimage.morphology import skeletonize
    g = np.asarray(Image.open(path).convert("L"))
    sk = skeletonize(g < 128)
    if skel_dilate > 0:
        from scipy.ndimage import binary_dilation, generate_binary_structure
        sk = binary_dilation(sk, generate_binary_structure(2, 2), iterations=int(skel_dilate))
    return _to_img(np.where(sk, 0, 255).astype(np.float32))


def _tf_canny(path, skel_dilate=0):
    from PIL import Image
    g = np.asarray(Image.open(path).convert("L"))
    try:
        import cv2
        ed = cv2.Canny(g, 50, 150)
    except Exception:
        from skimage.feature import canny as _c
        ed = (_c(g, sigma=1.5) * 255).astype(np.uint8)
    return _to_img(np.where(ed > 0, 0, 255).astype(np.float32))


_TRANSFORMS = {"gray": _tf_gray, "skel": _tf_skel, "canny": _tf_canny}


class _CsvImageDataset(torch.utils.data.Dataset):
    def __init__(self, rows, transform, skel_dilate=0):
        self.rows = rows              # [(abs_path, img_id), ...]
        self.tf = _TRANSFORMS[transform]
        self.d = int(skel_dilate)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        p, iid = self.rows[i]
        try:
            x = self.tf(p, self.d)
        except Exception:
            x = np.zeros((3, 256, 256), np.float32)
        return i, iid, torch.from_numpy(x)


def _collate(b):
    idx = torch.tensor([x[0] for x in b], dtype=torch.long)
    ids = torch.tensor([x[1] for x in b], dtype=torch.long)
    xs = torch.stack([x[2] for x in b], 0)
    return idx, ids, xs


# --------------------------------------------------------------------------- #
# VAE
# --------------------------------------------------------------------------- #
class VAEIO:
    def __init__(self, vae_path="pretrained_models/sd-vae-ft-ema", device="cuda",
                 encode_dtype=torch.bfloat16, decode_dtype=torch.float32, compile=False):
        from diffusers import AutoencoderKL
        self.device = torch.device(device)
        self.cpu = self.device.type == "cpu"
        self.vae = AutoencoderKL.from_pretrained(vae_path).to(self.device).eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)
        if not self.cpu:
            self.vae.to(memory_format=torch.channels_last)
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
        self.ed = torch.float32 if self.cpu else encode_dtype
        self.dd = torch.float32 if self.cpu else decode_dtype
        self.enc = torch.compile(self.vae.encoder) if compile else self.vae.encoder
        self.dec = torch.compile(self.vae.decoder) if compile else self.vae.decoder

    def _autocast(self, dtype):
        if self.cpu or dtype == torch.float32:
            return torch.autocast("cpu", enabled=False)
        return torch.autocast("cuda", dtype=dtype)

    @torch.inference_mode()
    def encode(self, x):
        """x: (B,3,256,256) CPU/GPU [-1,1] -> np.float16 (B,4,32,32) * SCALING."""
        x = x.to(self.device, non_blocking=True)
        if not self.cpu:
            x = x.contiguous(memory_format=torch.channels_last)
        with self._autocast(self.ed):
            h = self.vae.quant_conv(self.enc(x))
        h = h.float()
        return (h[:, : h.shape[1] // 2] * SCALING).half().cpu().numpy()

    @torch.inference_mode()
    def decode(self, z):
        """z: (B,4,32,32) CPU/GPU (scaled) -> torch.float32 (B,3,256,256) [-1,1] CPU."""
        z = z.to(self.device, non_blocking=True).float()
        if not self.cpu:
            z = z.contiguous(memory_format=torch.channels_last)
        with self._autocast(self.dd):
            img = self.dec(self.vae.post_quant_conv(z))
        return img.float().clamp(-1, 1).cpu()


# --------------------------------------------------------------------------- #
# shard 构建
# --------------------------------------------------------------------------- #
def _parse_rows(csv_path, limit=0, missing_ok=False):
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = r["image_path"]
            m = re.search(r"(\d+)\.png", p)
            if not m:
                continue
            full = p if os.path.isabs(p) else os.path.join(ROOT, p)
            if not os.path.isfile(full):
                if missing_ok:
                    continue
                raise FileNotFoundError(full)
            rows.append((full, int(m.group(1))))
            if limit and len(rows) >= limit:
                break
    return rows


def encode_csv(csv_path, out_dir, transform="gray", skel_dilate=0, vae_path="pretrained_models/sd-vae-ft-ema",
               batch=64, shard_size=5000, workers=12, device="cuda", limit=0, compile=False):
    rows = _parse_rows(csv_path, limit)
    ds = _CsvImageDataset(rows, transform, skel_dilate)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers,
                                     pin_memory=True, collate_fn=_collate, drop_last=False,
                                     persistent_workers=(workers > 0))
    vae = VAEIO(vae_path, device=device, compile=compile)
    os.makedirs(out_dir, exist_ok=True)
    buf_lat, buf_id, shard, done, t0 = [], [], 0, 0, time.time()
    total = len(rows)

    def flush():
        nonlocal buf_lat, buf_id, shard
        if not buf_lat:
            return
        np.savez_compressed(os.path.join(out_dir, f"shard_{shard:05d}.npz"),
                            latents=np.stack(buf_lat).astype(np.float16),
                            img_ids=np.array(buf_id, dtype=np.int64))
        shard += 1
        buf_lat, buf_id = [], []

    for _, ids, xs in dl:
        lat = vae.encode(xs)
        for k in range(lat.shape[0]):
            buf_lat.append(lat[k])
            buf_id.append(int(ids[k]))
            done += 1
        if len(buf_lat) >= shard_size:
            flush()
        if done % 2000 < batch or done == total:
            el = max(time.time() - t0, 1e-9)
            gpu = (torch.cuda.memory_allocated() / 1e9) if device != "cpu" else 0.0
            print(f"  {done}/{total}  {done/el:.0f} img/s  gpu={gpu:.2f}G", flush=True)
    flush()
    print(f"[done] {done} latents, {shard} shards -> {out_dir}/  ({time.time()-t0:.0f}s)", flush=True)
    return done


def bench(batch=64, n=512, vae_path="pretrained_models/sd-vae-ft-ema", device="cuda",
          encode_batch=None, decode_batch=None) -> None:
    vae = VAEIO(vae_path, device=device)
    x = torch.randn(n, 3, 256, 256)
    z = torch.randn(n, 4, 32, 32)
    eb = int(encode_batch or batch)
    db = int(decode_batch or batch)

    def _sync():
        if device != "cpu":
            torch.cuda.synchronize()

    for tag, fn, arg, bs in (("encode", vae.encode, x, eb), ("decode", vae.decode, z, db)):
        fn(arg[:bs]); _sync(); t0 = time.time()
        for s in range(0, n, bs):
            fn(arg[s:s + bs])
        _sync(); dt = time.time() - t0
        peak = (torch.cuda.max_memory_allocated() / 1e9) if device != "cpu" else 0.0
        print(f"  {tag}(bs={bs}): {n/dt:.1f} img/s ({dt:.2f}s for {n}), peak={peak:.2f}G")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--out")
    ap.add_argument("--transform", choices=list(_TRANSFORMS), default="gray")
    ap.add_argument("--skel-dilate", type=int, default=0)
    ap.add_argument("--vae-path", default="pretrained_models/sd-vae-ft-ema")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--shard-size", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--encode-batch", type=int, default=0)
    ap.add_argument("--decode-batch", type=int, default=0)
    a = ap.parse_args()
    if a.bench:
        bench(a.batch, 512, a.vae_path, a.device, a.encode_batch or None, a.decode_batch or None)
        return
    if not a.csv or not a.out:
        ap.error("--csv and --out required (or use --bench)")
    encode_csv(a.csv, a.out, a.transform, a.skel_dilate, a.vae_path, a.batch,
               a.shard_size, a.workers, a.device, a.limit, a.compile)


if __name__ == "__main__":
    main()
