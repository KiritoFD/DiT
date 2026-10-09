# -*- coding: utf-8 -*-
"""extract_dino_cls_raw.py — raw 全集 (393,486 张) 抽 DINOv2 CLS 特征。

朴素实现: 顺序分块 -> 多线程读图 -> 拼批上卡 -> 前向 -> 拷回预分配数组。
显存平坦 (模型 + 一个批次的激活), 不累积任何 GPU 张量。

口径与 assets/dino_feat_top10_g.npz 一致: CLS token, 256x256, 灰度->3通道,
ImageNet norm, fp32。输出 keys: feat / ids / calligs / scripts / glyph_ids

修过的 bug (2026-10-07): 原来 load() 写的是
    t = torch.from_numpy(g).float()[None, None].repeat(3, 1, 1)   # (1,1,H,W)->(3,1,H,W) 错
    t = F.interpolate(t[None], ...)                               # 5D 输入 -> 抛错
    except Exception: return torch.zeros(3, 256, 256)             # 静默吞成"全黑图"
  -> DINO 对全黑图输出同一常数 -> 39w 特征全同 -> 所有下游必然学不动。
  现在: 维度写对 ([None].repeat(3,1,1)), 异常直接抛出, 存盘前做有效性自检。

用法:
  PYTHONPATH=. python tools/extract_dino_cls_raw.py --batch 512 --workers 32
"""
import argparse
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir(os.environ.get("DIT_ROOT", "/root/Workspace/xy/DiT"))

SIZE = 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/root/Workspace/xy/UNIFIED_RAW/meta/train_clean.csv")
    ap.add_argument("--img-root", default="/root/Workspace/xy/UNIFIED_RAW")
    ap.add_argument("--out", default="assets/dino_feat_raw.npz")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help=">0 = 只跑前 N 张 (冒烟)")
    a = ap.parse_args()

    from src.loss.losses import _default_dino_ckpt, _load_local_dinov2
    dev = "cuda"
    ckpt = _default_dino_ckpt()
    model = _load_local_dinov2(ckpt).to(dev).eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"[dino] {ckpt}  batch={a.batch} workers={a.workers} -> {a.out}", flush=True)

    rows = []
    with open(a.csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((r["image_path"], int(r["img_id"]), int(r["calligrapher_id"]),
                         int(r["script_id"]), int(r["character_id"])))
    if a.limit:
        rows = rows[:a.limit]
    n = len(rows)
    print(f"[data] {n:,} 行  (callig={len({r[2] for r in rows})} "
          f"script={len({r[3] for r in rows})} char={len({r[4] for r in rows})})", flush=True)

    ids = np.array([r[1] for r in rows], dtype=np.int64)
    cal = np.array([r[2] for r in rows], dtype=np.int64)
    scr = np.array([r[3] for r in rows], dtype=np.int64)
    chs = np.array([r[4] for r in rows], dtype=np.int64)
    paths = [r[0] if os.path.isabs(r[0]) else os.path.join(a.img_root, r[0]) for r in rows]

    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    def load(p):
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if g is None:
            raise RuntimeError(f"读图失败: {p}")
        if g.shape[0] != SIZE or g.shape[1] != SIZE:
            g = cv2.resize(g, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(np.ascontiguousarray(g)).unsqueeze(0).repeat(3, 1, 1)

    pool = ThreadPoolExecutor(a.workers)
    feat = np.zeros((n, 384), np.float32)
    t0 = time.time()
    with torch.no_grad():
        for k in range(0, n, a.batch):
            chunk = paths[k:k + a.batch]
            imgs = list(pool.map(load, chunk))
            x = torch.stack(imgs).to(dev).float().div_(255.0)
            x = (x - mean) / std
            out = model(x)
            cls = out["x_norm_clstoken"] if (isinstance(out, dict)
                                             and "x_norm_clstoken" in out) \
                else out.last_hidden_state[:, 0, :]
            feat[k:k + len(chunk)] = cls.float().cpu().numpy()
            done = min(k + a.batch, n)
            if (k // a.batch) % 20 == 0:
                el = time.time() - t0
                print(f"  {done:>7,}/{n:,}  {done/el:5.0f} img/s  已 {el/60:.1f}min  "
                      f"ETA {el/done*(n-done)/60:.1f}min  "
                      f"显存 {torch.cuda.memory_allocated()/1e9:.2f}G", flush=True)
    pool.shutdown()

    _std = float(feat[:2000].std(0).mean())
    _uniq = len(np.unique(feat[:200], axis=0))
    print(f"[check] 逐维 std 均值={_std:.4f}  前200行不同数={_uniq}/200", flush=True)
    if _std < 1e-6 or _uniq <= 2:
        raise SystemExit("[FATAL] 特征退化 (常数/重复), 拒绝保存")
    np.savez(a.out, feat=feat, ids=ids, calligs=cal, scripts=scr, glyph_ids=chs)
    print(f"[done] {feat.shape} -> {a.out}  "
          f"({time.time()-t0:.0f}s, {n/(time.time()-t0):.0f} img/s)", flush=True)


if __name__ == "__main__":
    main()
