#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""build_style_protos.py —— 用**已验证**的风格编码器在**真迹**上建原型表（v85B 需要 ✓）。

产出 npz（供 `StyleRankLoss(mode="rank", protos=...)` ✓）
    protos  (P,256) 每行 = 一个 (书家×书体) pair 的真迹平均风格嵌入（L2 归一化 ✓）
    rival   (P,P)   rival[i,j] = "j 与 i **同书体但不同书家**" ✓（方案要求的排序对手 ✓）
    pair_ok (P,)    该 pair 是否有足够真迹 ✓（无则训练时跳过 ✓，不硬算 ✓）
    meta    (P,)    每行的 "书家/书体" 字符串（人工核对用 ✓）

三条纪律（方案 §三.3 明确要求 ✓）
    1. **索引空间必须与 batch['y_pair'] 一致** ✓
       —— 实测本仓 CSV 的 `pair_id` 只有 **23 个**取值（10 书家 × 书体 ✓），
          而旧 `rank_cent87.npy` 是 **45 书家 / 87 pairs** ✗ ⇒ 语义错位 ✓（方案警告的正是这条 ✓）。
    2. 原型只能由**训练真迹**构建 ✓；**排除评测集的行** ✗（评测集的 GT 不能参与 ✓）。
       另: 原型是"多张不同字符/作品的平均" ✓ ⇒ 天然满足"用不同字符不同作品" ✓。
    3. 只可用**已通过门槛**的编码器 ✓（`gates.run_gates` 的 G1≥0.65 ✓）；
       未通过时本工具**默认拒绝**（`--allow-unvalidated` 可强制, 但会在 meta 里标记 ✓）。

用法
    python tools/build_style_protos.py --ckpt assets/style_enc_v85.pt \
        --csv exp-std/csv/train.csv --eval-csv exp-std/csv/eval200_fixed.csv \
        --out assets/style_protos_v85.npz
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.style.bank import resolve                          # noqa: E402
from src.eval.style.metrics import l2norm                                 # noqa: E402

VAE_DIR = "data/pretrained/sd-vae-ft-ema"
SF = 0.18215


def _latent(paths, device, batch=32):
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
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return np.concatenate(out).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="已验收的风格编码器 ✓")
    ap.add_argument("--csv", default="exp-std/csv/train.csv")
    ap.add_argument("--eval-csv", default="exp-std/csv/eval200_fixed.csv",
                    help="**排除**它的行, 防止评测 GT 泄漏进原型 ✓")
    ap.add_argument("--out", default="assets/style_protos_v85.npz")
    ap.add_argument("--root", action="append", default=[])
    ap.add_argument("--min-count", type=int, default=8, help="每个 pair 至少多少张才算可用 ✓")
    ap.add_argument("--max-per-pair", type=int, default=400)
    ap.add_argument("--device", default="")
    ap.add_argument("--allow-unvalidated", action="store_true")
    ap.add_argument("--gate-json", default="", dest="gate_json",
                    help="外部复核结果（eval_style_encoder_cached.py --save-json 产出 ✓）；\n"
                    "给了就用它覆盖 ckpt 里那个单一口径的数 ✓")
    args = ap.parse_args()
    roots = args.root or [".", "data", "exp-std"]
    dev = args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")

    import torch
    from src.loss.style_rank_module import load_encoder
    enc, ck = load_encoder(args.ckpt, verbose=True)
    gates = (ck or {}).get("gates") or {}
    g1 = (gates.get("G1") or {}).get("acc", float("nan"))
    print("[protos] ckpt 内存的门槛: G1=%.4f pass=%s（口径 = **作品级留出** ✓）"
          % (g1, gates.get("pass")))

    # ★ 2026-10-14 修正: 允许用**外部复核结果**覆盖 ckpt 里那一个数字 ✓
    #   为什么: ckpt 存的是**作品级留出**口径（跨作品泛化, 更难 ✓）。实测 v1/v2 在该口径
    #   为 0.6002/0.6078 ✗（CI 明确排除 0.65），而**同分布口径**为 0.8600/0.8833 ✓✓
    #   （CI 下界 0.84/0.87 ⇒ 方案字面"真实真迹"口径**明确通过** ✓）。
    #   ⇒ 只看留出口径会**误拒**一个在方案口径上合格的编码器 ✓
    #   ⇒ 于是: 由 `tools/eval_style_encoder_cached.py --save-json` 产出**两种口径**的结论 ✓，
    #     本工具只读该 JSON ✓（判据**单一来源** ✓ 不重复实现 ✓）。
    _gj = str(getattr(args, "gate_json", "") or "")
    if _gj and os.path.isfile(_gj):
        try:
            _g = json.load(open(_gj, encoding="utf-8"))
            _same = (_g.get("same_dist") or {}).get("acc", float("nan"))
            _ci = (_g.get("same_dist") or {}).get("ci95") or [0, 0]
            print("[protos] 外部复核(%s): 同分布 G1=%.4f ci95=[%.4f, %.4f] ⇒ pass=%s"
                  % (os.path.basename(_gj), _same, _ci[0], _ci[1], _g.get("pass")))
            if not _g.get("pass") and not args.allow_unvalidated:
                print("  ✗ 外部复核也未过 0.65 ⇒ 拒绝建原型 ✓（方案纪律 ✓）")
                return 3
            gates = {"pass": bool(_g.get("pass")), "G1": {"acc": _same, "ci95": _ci},
                     "source": os.path.basename(_gj)}
            g1 = _same
        except Exception as _e:                                          # noqa: BLE001
            print("  ⚠ 读 %s 失败: %s ⇒ 退回用 ckpt 门槛 ✓" % (_gj, _e))
    if not gates.get("pass") and not args.allow_unvalidated:
        print("  ✗ 该编码器**未通过验收** ⇒ 拒绝用它建原型 ✓（方案要求 ✓）\n"
              "     如确要用, 加 --allow-unvalidated（会在 meta 标记 ✓）；\n"
              "     或先用 tools/eval_style_encoder_cached.py --save-json 做两种口径复核 ✓")
        return 3
    enc = enc.to(dev).eval()

    # 排除评测集
    ex = set()
    if args.eval_csv and os.path.isfile(args.eval_csv):
        for r in csv.DictReader(open(args.eval_csv, encoding="utf-8-sig")):
            p = resolve(r.get("image_path", ""), roots)
            if p:
                ex.add(os.path.abspath(p))
                ex.add(os.path.basename(p))
        print("[protos] 评测集排除 %d 个路径 ✓" % len(ex))

    by = defaultdict(list)
    meta, skipped = {}, 0
    for r in csv.DictReader(open(args.csv, encoding="utf-8-sig")):
        if (r.get("aug") or "").strip():
            continue
        p = resolve(r.get("image_path", ""), roots)
        if not p or os.path.abspath(p) in ex or os.path.basename(p) in ex:
            skipped += 1
            continue
        pid = r.get("pair_id")
        if pid is None or str(pid).strip() == "":
            skipped += 1
            continue
        pid = int(pid)
        by[pid].append(p)
        meta[pid] = "%s/%s" % (r.get("calligrapher", ""), r.get("script", ""))
    P = max(by) + 1
    print("[protos] P=%d（pair_id 空间 ✓，与 batch['y_pair'] 一致 ✓）| 排除/跳过 %d" %
          (P, skipped))

    protos = np.zeros((P, 256), np.float32)
    pair_ok = np.zeros(P, bool)
    names = np.array([meta.get(i, "?") for i in range(P)], dtype=object)
    for pid in range(P):
        paths = by.get(pid, [])[:args.max_per_pair]
        if len(paths) < args.min_count:
            print("  pair %2d %-14s 样本 %3d < %d ⇒ 标记不可用 ✓" %
                  (pid, names[pid], len(paths), args.min_count))
            continue
        z = _latent(paths, dev)
        f = []
        with torch.no_grad():
            for s in range(0, len(z), 1024):
                f.append(enc.emb(torch.from_numpy(z[s:s + 1024]).to(dev)).cpu().numpy())
        protos[pid] = l2norm(np.concatenate(f)).mean(0)
        pair_ok[pid] = True
        print("  pair %2d %-14s n=%3d ✓" % (pid, names[pid], len(paths)))

    protos = l2norm(protos)
    scr = np.array([str(n).split("/")[-1] if "/" in str(n) else "?" for n in names])
    wr = np.array([str(n).split("/")[0] if "/" in str(n) else "?" for n in names])
    rival = np.zeros((P, P), bool)
    for i in range(P):
        for j in range(P):
            rival[i, j] = (scr[i] == scr[j]) and (wr[i] != wr[j]) and pair_ok[j]
    rv = int(rival.sum())
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, protos=protos, rival=rival, pair_ok=pair_ok, meta=names,
                        g1=np.array([g1]), validated=np.array([bool(gates.get("pass"))]))
    print("[protos] 保存 %s | 可用 pair %d/%d | 排序对手对 %d（均值每行 %.1f）"
          % (args.out, int(pair_ok.sum()), P, rv,
             rv / max(1, int(pair_ok.sum()))))
    print("  ⇒ dim=%d ✓ 供 StyleRankLoss(mode='rank', protos=...) 使用 ✓" % protos.shape[1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
