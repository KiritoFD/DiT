#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""style_encoder_accept.py —— 风格编码器**验收**（方案 §三.2 的三条门槛 ✓）。

判据实现全部在 `src/eval/style/gates.py` ✓（与训练脚本共用同一套 ✓，
避免"训练时好看、验收时过不了" ✗）。本文件只负责:
    ① 均衡读取真迹行（⚠ 按 CSV 顺序截断只会拿到前 1–2 位书家 ✗ —— 实测 max_img=2500 时
       只剩 3 位书家，而 G1 需要"异书家对" ⇒ 无从构造 ✓）
    ② 把图像/ latent 编码成嵌入（`--enc` 走 timm 现成编码器 ✓ / `--ckpt` 走自训 latent 编码器 ✓）
    ③ 调 `run_gates` 并打印判定 ✓

用法
    python tools/style_encoder_accept.py --enc dinov2_vitl14 --max-img 2500     # 基线 ✓
    python tools/style_encoder_accept.py --ckpt assets/style_enc_v85.pt         # 自训 latent ✓
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.style.bank import resolve, work_id                          # noqa: E402
from src.eval.style.gates import run_gates                                # noqa: E402

VAE_DIR = "data/pretrained/sd-vae-ft-ema"
SF = 0.18215


def build_rows(csv_path, roots, max_img=0, drop_aug=True):
    """均衡抽样（按 (书家,书体) 槽位轮转 ✓）—— 直接截断会只剩 1–2 位书家 ✗。"""
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8-sig")))
    by, miss = defaultdict(list), 0
    for r in rows:
        if drop_aug and (r.get("aug") or "").strip():
            continue
        p = resolve(r.get("image_path", ""), roots)
        if not p:
            miss += 1
            continue
        d = {"path": p, "writer": (r.get("calligrapher") or "").strip(),
             "script": (r.get("script") or "").strip(),
             "char": (r.get("character") or "").strip(),
             "work": work_id(r.get("src_image_path", ""), r.get("image_path", "")),
             "std": resolve(r.get("std_path", ""), roots) or ""}
        by[(d["writer"], d["script"])].append(d)
    if not max_img:
        out = [d for k in sorted(by) for d in by[k]]
    else:
        out, i = [], 0
        while len(out) < max_img:
            added = False
            for k in sorted(by):
                if i < len(by[k]):
                    out.append(by[k][i])
                    added = True
                    if len(out) >= max_img:
                        break
            if not added:
                break
            i += 1
    print("[accept] 可用 %d（缺图 %d）| 书家 %d | 字 %d | 作品 %d | 槽位 %d"
          % (len(out), miss, len({r['writer'] for r in out}), len({r['char'] for r in out}),
             len({r['work'] for r in out}), len(by)))
    return out


def _latent_of(paths, device, batch=32):
    """像素 -> v84A 同一 VAE 契约的 latent ✓（自训编码器必须走这条路 ✓）。"""
    from PIL import Image
    from diffusers.models import AutoencoderKL
    import torch
    vae = AutoencoderKL.from_pretrained(VAE_DIR).to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            x = torch.stack([
                torch.from_numpy(np.asarray(Image.open(p).convert("RGB")))
                .permute(2, 0, 1).float() / 127.5 - 1.0 for p in paths[i:i + batch]
            ]).to(device)
            out.append((vae.encode(x).latent_dist.sample() * SF).cpu().numpy())
    del vae
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return np.concatenate(out).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="exp-std/csv/train.csv")
    ap.add_argument("--root", action="append", default=[])
    ap.add_argument("--max-img", type=int, default=2500)
    ap.add_argument("--enc", default="", help="timm 现成编码器（基线对照 ✓）")
    ap.add_argument("--pool", default="cls")
    ap.add_argument("--ckpt", default="", help="自训 latent 风格编码器权重 ✓")
    ap.add_argument("--device", default="")
    ap.add_argument("--n-pairs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    roots = args.root or [".", "data", "exp-std"]
    dev = args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")

    rows = build_rows(args.csv, roots, args.max_img)
    if not rows:
        print("无可用图像 ✗")
        return 2

    if args.enc:
        from src.eval.style.encoders import TimmEncoder
        enc = TimmEncoder.load(args.enc, pool=args.pool, device=dev)
        emb = enc.embed_paths([r["path"] for r in rows], batch=24, progress=True)[0]
        name = "%s(%s)" % (args.enc, args.pool)
        emb_std = None
        std_rows = [{"path": r["std"], "writer": r["writer"], "script": r["script"],
                     "char": r["char"], "work": "std"} for r in rows if r["std"]]
        if std_rows:
            emb_std = enc.embed_paths([r["path"] for r in std_rows], batch=24)[0]
    elif args.ckpt:
        import torch
        from tools.train_style_encoder import Enc
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        model = Enc(len(ck["calls"]), d=ck.get("d", 256), ch=ck.get("ch", 96)).to(dev).eval()
        model.load_state_dict(ck["model"])
        name = os.path.basename(args.ckpt)

        def _emb(paths):
            lat = _latent_of(paths, dev)
            out = []
            with torch.no_grad():
                for s in range(0, len(lat), 1024):
                    out.append(model.emb(
                        torch.from_numpy(lat[s:s + 1024]).to(dev)).cpu().numpy())
            return np.concatenate(out)

        emb = _emb([r["path"] for r in rows])
        std_rows = [{"path": r["std"], "writer": r["writer"], "script": r["script"],
                     "char": r["char"], "work": "std"} for r in rows if r["std"]]
        emb_std = _emb([r["path"] for r in std_rows]) if std_rows else None
    else:
        print("要么 --enc（基线 ✓），要么 --ckpt（自训 ✓）")
        return 2

    print("\n===== 验收: %s (n=%d) =====" % (name, len(rows)))
    res = run_gates(emb, rows, emb_std=emb_std, rows_std=std_rows or None,
                    n_pairs=args.n_pairs, seed=args.seed)
    print("\n  判定: G1 %s | G2 %s | G3（需与真迹对照后判 ✓）"
          % ("✓ 过" if res["G1"]["acc"] >= 0.65 else "✗ 不过",
             "✓ 过" if (res["G2"]["map"] == res["G2"]["map"]
                        and res["G2"]["map"] > res["G2"]["base_script"] + 0.05) else "✗ 不过"))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
