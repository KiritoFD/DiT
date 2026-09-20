# -*- coding: utf-8 -*-
"""extract_dino_cls_base.py — base 全量 DINO v2 CLS 特征提取 (GPU 独占).

输出: assets/dino_cls_base.npz {"feat": (M,384), "calligs": raw calligrapher_id 数组}
与 remote_dino_extract.py 同口径: CLS token (last_hidden_state[:,0]), 256 input,
ImageNet norm, fp16, batch 1024.
"""
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir("/root/Workspace/xy/DiT")

BATCH = 1024
DINO_SIZE = 256


def main():
    from src.loss.losses import _default_dino_ckpt, _load_local_dinov2
    dev = "cuda"
    ckpt = _default_dino_ckpt()
    model = _load_local_dinov2(ckpt).to(dev).eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"[dino] loaded {ckpt}", flush=True)

    import cv2
    from concurrent.futures import ThreadPoolExecutor
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev, dtype=torch.float16).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev, dtype=torch.float16).view(1, 3, 1, 1)

    rows = []
    for r in csv.DictReader(open("assets/train_base_noaug.csv", encoding="utf-8")):
        rows.append((r["image_path"], int(r["calligrapher_id"])))
    print(f"rows: {len(rows)}", flush=True)

    def load(path):
        try:
            g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if g is None:
                g = np.zeros((256, 256), np.uint8)
            t = torch.from_numpy(g).float()[None, None].repeat(3, 1, 1)
            t = F.interpolate(t[None], size=(DINO_SIZE, DINO_SIZE), mode="bicubic",
                              align_corners=False)[0]
            # ★ 2026-09-19 修: 同 extract_dino_cls_50k.py —— cv2 灰度 0..255, 缺 /255
            #   会让 ImageNet-norm 后输入幅度 ~1e3, ViT attention 饱和, CLS 全行恒定
            t = t / 255.0
            return t
        except Exception:
            return torch.zeros(3, DINO_SIZE, DINO_SIZE)

    pool = ThreadPoolExecutor(16)
    feats, calligs = [], []
    t0 = time.time()
    with torch.no_grad():
        for k in range(0, len(rows), BATCH):
            chunk = rows[k:k + BATCH]
            imgs = [x for x in pool.map(lambda r: load(r[0]), chunk) if x is not None]
            x = torch.stack(imgs).to(dev).float()
            x = (x - mean.float()) / std.float()
            out = model(x)
            if isinstance(out, dict) and "x_norm_clstoken" in out:
                cls = out["x_norm_clstoken"]
            else:
                cls = out.last_hidden_state[:, 0, :]
            feats.append(cls.float().cpu().numpy())
            calligs.extend(int(r[1]) for r in chunk)
            if (k // BATCH) % 10 == 0:
                print(f"  {k+len(chunk)}/{len(rows)} ({k+len(chunk)}/{time.time()-t0:.0f}s)",
                      flush=True)
    feat = np.concatenate(feats)
    np.savez("assets/dino_cls_base.npz",
             feat=feat.astype(np.float32), calligs=np.array(calligs, dtype=np.int64))
    print(f"[done] {feat.shape} -> assets/dino_cls_base.npz", flush=True)


if __name__ == "__main__":
    main()
