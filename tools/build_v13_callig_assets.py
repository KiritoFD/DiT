# -*- coding: utf-8 -*-
"""为 50k 数据集生成 v13 的书家资产：callig_id_map + 预训练书家表。

## 为什么需要
新数据集有 **45** 个书家（raw id 53..9022），而 v12 用的是 base 的 52 个。
两件事都要重做：
  1. `callig_id_map`：把 raw id 映射到 0..44（`--callig-id-map` 会据此设
     `num_calligraphers`）
  2. `callig_emb_pretrained`：训练侧有**严格断言**
     `_emb.shape == (num_calligraphers, dim)` —— 形状不对会直接 AssertionError。

## 预训练表怎么来
45 个里 **35 个在 base 52 里**（可直接取其行做 warm start），
**10 个是新增的**（raw id 9011..9022）—— 这 10 行**随机初始化**，
与"新书家从零学"一致；其余保持预训练值。
（也可以改用 tools/pretrain_callig_emb_base.py 全量重训，但那需要先有
  对应的 dino_cls npz，成本高且语义上没必要。）

用法:
    python tools/build_v13_callig_assets.py \
        --csv assets/train_50k.csv \
        --base-map assets/callig_id_map_base.json \
        --base-emb assets/callig_emb_pretrained_base.pt \
        --out-map assets/callig_id_map_50k.json \
        --out-emb assets/callig_emb_pretrained_50k.pt
"""
import argparse
import csv
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.utils.callig_map import build_callig_id_map, save_callig_id_map  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="assets/train_50k.csv")
    ap.add_argument("--base-map", default="assets/callig_id_map_base.json")
    ap.add_argument("--base-emb", default="assets/callig_emb_pretrained_base.pt")
    ap.add_argument("--out-map", default="assets/callig_id_map_50k.json")
    ap.add_argument("--out-emb", default="assets/callig_emb_pretrained_50k.pt")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    # ── 1. 新 callig_id_map（按 raw id 排序 -> 0..N-1，与 build_callig_map 同规则）──
    mapping = build_callig_id_map([a.csv])
    save_callig_id_map(mapping, a.out_map)
    n = len(mapping)
    print(f"[callig] 词表 {n} 个 -> {a.out_map}", flush=True)

    # ── 2. 预训练表：能对上的取 base 行，对不上的随机 ──────────────────────
    bm = json.load(open(a.base_map, encoding="utf-8"))
    base_map = {int(k): int(v) for k, v in bm["id_map"].items()}
    _d = torch.load(a.base_emb, map_location="cpu", weights_only=False)
    base_emb = _d["embedding"] if isinstance(_d, dict) else _d
    print(f"[callig] base 表 {tuple(base_emb.shape)}", flush=True)

    g = torch.Generator().manual_seed(a.seed)
    out = torch.empty(n, a.dim)
    out.normal_(0, 0.02, generator=g)          # 默认随机（新书家）
    copied, fresh = [], []
    for raw, idx in sorted(mapping.items()):
        if raw in base_map:
            out[idx] = base_emb[base_map[raw]].float()
            copied.append(raw)
        else:
            fresh.append(raw)
    print(f"[callig] 从 base 复制 {len(copied)} 个；新增随机初始化 {len(fresh)} 个 "
          f"{fresh[:12]}", flush=True)

    assert out.shape == (n, a.dim), out.shape
    torch.save({"embedding": out}, a.out_emb)
    print(f"[callig] 预训练表 {tuple(out.shape)} -> {a.out_emb}", flush=True)

    # 打印映射，便于人工核对
    names = {}
    for r in csv.DictReader(open(a.csv, encoding="utf-8")):
        names.setdefault(int(r["calligrapher_id"]), r["calligrapher"])
    for raw, idx in sorted(mapping.items()):
        tag = "base" if raw in base_map else "NEW "
        print(f"    {tag} raw={raw:>5} -> idx={idx:>3}  {names.get(raw, '?')}")


if __name__ == "__main__":
    main()
