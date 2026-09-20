# -*- coding: utf-8 -*-
"""extract_dino_cls_50k.py — 50k 数据集 DINO v2 CLS 特征提取 (带 script 标签).

Stage 1 的前置: (书家×书体) 联合风格表的 SupCon 预训练需要**每行 DINO 特征对应的
(calligrapher_id, script_id)**。base 版 `dino_cls_base.npz` 只有 `calligs`、且是 base
数据集(54892 行), 不含 script、也不是 50k 的图 -> 必须为 50k 重新提取。

与 extract_dino_cls_base.py 同口径(CLS token, 256 input, ImageNet norm, fp16, batch 1024),
额外保存 `scripts` 与 `img_ids`, 供 pretrain_callig_script_emb.py 建每对质心。

输出: assets/dino_cls_50k.npz
    {"feat": (M,384) f32, "calligs": (M,) i64, "scripts": (M,) i64, "img_ids": (M,) i64}

用法(远端 GPU, 一次性, ~1 分钟):
    python tools/extract_dino_cls_50k.py --csv assets/train_50k_v2.csv \
        --out assets/dino_cls_50k.npz
"""
import argparse
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="assets/train_50k_v2.csv")
    ap.add_argument("--out", default="assets/dino_cls_50k.npz")
    a = ap.parse_args()

    from src.loss.losses import _default_dino_ckpt, _load_local_dinov2
    dev = "cuda"
    ckpt = _default_dino_ckpt()
    model = _load_local_dinov2(ckpt).to(dev).eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"[dino] loaded {ckpt}", flush=True)

    import cv2
    from concurrent.futures import ThreadPoolExecutor
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    rows = []
    for r in csv.DictReader(open(a.csv, encoding="utf-8")):
        # img_id: 优先显式列, 回退文件名(与 latent_dataset.extract_img_id 同思路)
        iid = r.get("img_id") or r.get("old_50k_id") or ""
        iid = int(iid) if str(iid).strip() else int(os.path.splitext(
            os.path.basename(r["image_path"]))[0])
        rows.append((r["image_path"], int(r["calligrapher_id"]),
                     int(r["script_id"]), iid))
    print(f"rows: {len(rows)}", flush=True)

    def load(path):
        """灰度图 -> (3,256,256) 0..1 (ImageNet norm 前的口径)。

        ★ 2026-09-19 两个叠加的 bug (都已修, 二者共同造成"全库特征恒定"):
          1. `(1,1,H,W).repeat(3,1,1)` 参数数 < 维数 -> **每张图都 RuntimeError**,
             而下方 `except: return zeros` 把它吞掉 -> 所有图变成同一张全零图,
             CLS 全行恒定 (unique=1), SupCon 的 DINO 锚定全程失效。
             教训: 用来兜底的 except 必须至少打日志/计数, 否则静默全灭。
          2. cv2 灰度是 0..255, 缺 /255 -> (修复 1 后) 输入幅度 ~1e3,
             ViT attention 饱和, CLS 仍近恒定。
        """
        try:
            g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if g is None:
                raise ValueError(f"imread failed: {path}")
            t = torch.from_numpy(g).float()[None, None]              # (1,1,H,W)
            t = F.interpolate(t, size=(DINO_SIZE, DINO_SIZE), mode="bicubic",
                              align_corners=False)                   # (1,1,256,256)
            t = (t / 255.0).repeat(1, 3, 1, 1)                       # (1,3,256,256)
            return t[0]
        except Exception as e:
            print(f"[load] ✗ {path}: {e}", flush=True)
            return None

    pool = ThreadPoolExecutor(16)
    feats, calligs, scripts, img_ids = [], [], [], []
    n_fail = 0
    t0 = time.time()
    with torch.no_grad():
        for k in range(0, len(rows), BATCH):
            chunk = rows[k:k + BATCH]
            # ⚠ feat 与标签必须逐行对齐: load 失败的行连同行标签一起丢弃
            loaded = [(r, img) for r, img in zip(chunk, pool.map(lambda r: load(r[0]), chunk))
                      if img is not None]
            n_fail += len(chunk) - len(loaded)
            if not loaded:
                continue
            ok_rows, imgs = zip(*loaded)
            x = torch.stack(list(imgs)).to(dev)
            x = (x - mean) / std
            out = model(x)
            cls = (out["x_norm_clstoken"] if isinstance(out, dict) and "x_norm_clstoken" in out
                   else out.last_hidden_state[:, 0, :])
            feats.append(cls.float().cpu().numpy())
            calligs.extend(int(r[1]) for r in ok_rows)
            scripts.extend(int(r[2]) for r in ok_rows)
            img_ids.extend(int(r[3]) for r in ok_rows)
            if (k // BATCH) % 10 == 0:
                print(f"  {k+len(chunk)}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)
    feat = np.concatenate(feats)
    _sub = feat[np.random.RandomState(0).choice(len(feat), min(2000, len(feat)), replace=False)]
    _n_uniq = len(np.unique(np.round(_sub, 4), axis=0))
    print(f"[check] 失败 {n_fail}/{len(rows)} 行; 特征唯一值(采样 {_sub.shape[0]}): {_n_uniq}",
          flush=True)
    np.savez(a.out,
             feat=feat.astype(np.float32),
             calligs=np.array(calligs, dtype=np.int64),
             scripts=np.array(scripts, dtype=np.int64),
             img_ids=np.array(img_ids, dtype=np.int64))
    print(f"[done] {feat.shape} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
