# -*- coding: utf-8 -*-
"""
build_dino_cache.py — REPA teacher (DINOv2) 特征离线预提取。

动机: 训练每个 step 对 batch GT 图实时跑 DINOv2 ViT-S/14 前向 (~40ms/step,
batch=192 时占整步 ~15-18% GPU 时间)。GT 图像集合固定 (csv 每行一个文件),
教师特征只依赖图像 → 本脚本一次性预提取全部特征, 训练时查表免前向。

用法 (远程 4090, ~10 分钟跑完 85k 张):
    /opt/conda/envs/cu121/bin/python tools/build_dino_cache.py \
        --csv assets/train_fame3_sym_full.csv \
        --img-root data/imgs/final_imgs_256 \
        --out data/dino_cache/v11_sym_full \
        --batch-size 128 --workers 8

输出 (cache_dir):
    feats.f16  — (N, 256, 384) float16 连续二进制 (np.memmap 可读)
    ids.npy    — (N,) int64 img_id (与 feats 行序一一对应, 顺序 = csv 行序)
    meta.json  — {"n", "patches", "dim", "backbone", "csv"}

预处理与 src/loss/losses.py:REPALoss._teacher_forward 严格一致:
[-1,1] -> [0,1] -> ImageNet 归一化 -> 224 bicubic -> x_norm_patchtokens。
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.loss.losses import _default_dino_ckpt, _load_local_dinov2, _TeacherWrapper

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
IMAGE_SIZE = 256
TEACHER_SIZE = 224


class _ImgDataset(Dataset):
    """csv 行序加载 256x256 GT 图 -> [-1,1] (3,256,256), 与 MCCDLatentDataset 一致。"""

    def __init__(self, rows, img_root):
        self.rows = rows
        self.img_root = img_root

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        p = self.rows[idx]["image_path"]
        full = p if os.path.isabs(p) else os.path.join(os.getcwd(), p)
        if not os.path.isfile(full) and self.img_root:
            full = os.path.join(self.img_root, f"{self._img_id(p)}.png")
        with Image.open(full) as im:
            img = im.convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0 * 2.0 - 1.0
        if arr.shape[-2:] != (IMAGE_SIZE, IMAGE_SIZE):
            arr = np.asarray(img.resize((IMAGE_SIZE, IMAGE_SIZE)), dtype=np.float32) / 255.0 * 2.0 - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1), self._img_id(p)

    @staticmethod
    def _img_id(p):
        import re
        m = re.search(r"(\d+)\.png", p)
        return int(m.group(1)) if m else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="training csv (row order = feature row order)")
    ap.add_argument("--img-root", default="data/imgs/final_imgs_256")
    ap.add_argument("--out", required=True, help="cache output dir")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--backbone", default="dinov2_vits14")
    ap.add_argument("--teacher-ckpt", default="",
                    help="local safetensors (default: auto-detect data/pretrained/dinov2_vits14_pretrain.safetensors)")
    ap.add_argument("--max-images", type=int, default=0, help="smoke test: only first N rows")
    args = ap.parse_args()

    with open(args.csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.max_images:
        rows = rows[:args.max_images]
    n = len(rows)
    print(f"[dino-cache] csv={args.csv} rows={n}")

    if os.path.exists(os.path.join(args.out, "meta.json")):
        print(f"[dino-cache] cache already exists in {args.out} — delete it to rebuild")
        return
    os.makedirs(args.out, exist_ok=True)

    # teacher: 与 REPALoss 相同的加载链 (本地 safetensors -> torch.hub)
    ckpt = args.teacher_ckpt or _default_dino_ckpt()
    teacher = None
    if ckpt and os.path.exists(ckpt):
        try:
            teacher = _load_local_dinov2(ckpt)
            print(f"[dino-cache] loaded local teacher from {ckpt}")
        except Exception as e:  # noqa: BLE001
            print(f"[dino-cache] local teacher failed ({e!r}); falling back to torch.hub")
    if teacher is None:
        teacher = torch.hub.load("facebookresearch/dinov2", args.backbone)
    if not isinstance(teacher, _TeacherWrapper):
        teacher = _TeacherWrapper(teacher)
    teacher = teacher.to(args.device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    ds = _ImgDataset(rows, args.img_root)
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers,
                    pin_memory=True, shuffle=False, drop_last=False)

    feats_path = os.path.join(args.out, "feats.f16")
    ids_path = os.path.join(args.out, "ids.npy")
    meta_path = os.path.join(args.out, "meta.json")

    patches = dim = None
    n_done = 0
    all_ids = []
    import time
    t0 = time.time()
    with open(feats_path, "ab") as ffeats:
        with torch.no_grad():
            for x, img_ids in dl:
                x = x.to(args.device, non_blocking=True)
                xn = ((x + 1.0) / 2.0 - MEAN.to(x.device)) / STD.to(x.device)
                x224 = F.interpolate(xn, size=(TEACHER_SIZE, TEACHER_SIZE),
                                     mode="bicubic", align_corners=False)
                out = teacher.forward_features(x224)
                feats = out.float().cpu().numpy().astype(np.float16)  # (B, P, D)
                if patches is None:
                    patches, dim = feats.shape[1], feats.shape[2]
                    print(f"[dino-cache] patch tokens: (N,{patches},{dim})")
                ffeats.write(feats.tobytes())
                all_ids.extend(int(i) for i in img_ids)
                n_done += x.shape[0]
                if n_done % (args.batch_size * 10) == 0 or n_done == n:
                    rate = n_done / max(1e-6, time.time() - t0)
                    eta = (n - n_done) / max(1e-6, rate)
                    print(f"[dino-cache] {n_done}/{n} ({rate:.0f} img/s, ETA {eta/60:.1f} min)")

    ids = np.asarray(all_ids, dtype=np.int64)
    np.save(ids_path, ids)

    meta = {"n": int(n), "patches": int(patches), "dim": int(dim),
            "backbone": args.backbone, "csv": args.csv}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[dino-cache] done: {n} imgs -> {args.out} "
          f"({os.path.getsize(feats_path) / 2**30:.1f} GiB feats, "
          f"{time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
