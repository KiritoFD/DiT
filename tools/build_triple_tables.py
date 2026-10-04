#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""build_triple_tables.py — 用跨视图 SupCon (v2 方法) 预训练三张独立条件表。

书家(10 类) / 书体(3 类) / 汉字(4677 类), 各一张 1024 维表, 供 v53 三表编码器
(LabelEmbedder x3 + concat + Linear) 作初始化。方法与 char_script_supcon v2
完全一致: view_A=表行, view_B=g(DINO feat), 正对=同类跨样本对 (CLIP 式双向)。

标签来源: exp-std/csv/train.csv (经 img_id 对齐 dino npz 行)。
产物: assets/triple_tables/{callig,font,char}_table.npy + *_index.json + char_remap.json
"""
import csv
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.chdir(os.environ.get("DIT_ROOT", "/root/Workspace/xy/DiT"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DIM = 1024
OUT = "assets/triple_tables"


def supcon_train(feat, labels, n_cls, dim, steps, lr, batch, k, temp, tag):
    lab = torch.from_numpy(labels.astype(np.int64)).to(DEV)
    feat_t = torch.from_numpy(feat.astype(np.float32)).to(DEV)
    feat_dim = feat_t.shape[1]
    W = nn.Embedding(n_cls, dim).to(DEV)
    nn.init.normal_(W.weight, std=0.02)
    P = nn.Sequential(nn.Linear(feat_dim, dim), nn.GELU(),
                      nn.Linear(dim, dim)).to(DEV)
    opt = torch.optim.Adam(list(W.parameters()) + list(P.parameters()), lr=lr)

    by = defaultdict(list)
    for i, g in enumerate(lab.tolist()):
        by[int(g)].append(i)
    multi = np.array(sorted(g for g, v in by.items() if len(v) >= 2), dtype=np.int64)
    print(f"[{tag}] 类别 {n_cls} (出现 {len(by)}), 可做正对 {len(multi)}")
    if len(multi) < 2:
        raise SystemExit(f"[FATAL] {tag}: 可做正对的类不足")
    C = max(1, batch // k)

    t0 = time.time()
    for step in range(1, steps + 1):
        sel = np.random.choice(multi, size=min(C, len(multi)), replace=False)
        idxs = []
        for g in sel:
            v = by[int(g)]
            idxs += list(v) if len(v) <= k else list(np.random.choice(v, size=k, replace=False))
        idx = torch.tensor(idxs, device=DEV, dtype=torch.long)
        y = lab[idx]
        zA = F.normalize(W(y), dim=-1)
        zB = F.normalize(P(feat_t[idx]), dim=-1)
        logits = zA @ zB.T / temp
        tgt = (y[:, None] == y[None, :]).float()
        la = -((logits.log_softmax(1) * tgt).sum(1) / tgt.sum(1).clamp_min(1)).mean()
        lb = -((logits.log_softmax(0) * tgt).sum(0) / tgt.sum(0).clamp_min(1)).mean()
        loss = 0.5 * (la + lb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 1 or step % 500 == 0 or step == steps:
            print(f"  [{tag}] step {step}: loss={loss.item():.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return W.weight.detach().cpu().numpy()


def main():
    import argparse
    os.makedirs(OUT, exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="assets/dino_feat_top10_g.npz")
    ap.add_argument("--csv", default="exp-std/csv/train.csv")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temp", type=float, default=0.07)
    a = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    ids = np.load("data/dino_cache/top10_v1/ids.npy")
    z = np.load(a.npz)
    feat = z["feat"].astype(np.float32)
    assert len(ids) == len(feat), (len(ids), len(feat))
    by_img = {}
    for r in csv.DictReader(open(a.csv, encoding="utf-8")):
        by_img[int(r["img_id"])] = r
    rows = [by_img[int(i)] for i in ids]
    print(f"[data] {len(rows)} 样本, feat_dim={feat.shape[1]}")

    # 词表必须覆盖 eval 集 (否则 dataset 在 eval 样本上 KeyError);
    # eval 独有的字/书家落在未训练行 (随机 init), SupCon 只训 train 出现的类。
    _eval_rows = []
    _eval_csv = "exp-std/csv/eval200_fixed.csv"
    if os.path.exists(_eval_csv):
        _eval_rows = list(csv.DictReader(open(_eval_csv, encoding="utf-8")))
        print(f"[data] eval 词表合并: +{len(_eval_rows)} 行")
    cal_ids = sorted({int(r["calligrapher_id"]) for r in rows}
                     | {int(r["calligrapher_id"]) for r in _eval_rows})
    fnt_ids = sorted({int(r["script_id"]) for r in rows}
                     | {int(r["script_id"]) for r in _eval_rows})
    hanzi = sorted({r["character"] for r in rows}
                   | {r["character"] for r in _eval_rows})
    hz_idx = {h: i for i, h in enumerate(hanzi)}
    cal_lab = np.array([cal_ids.index(int(r["calligrapher_id"])) for r in rows])
    fnt_lab = np.array([fnt_ids.index(int(r["script_id"])) for r in rows])
    chr_lab = np.array([hz_idx[r["character"]] for r in rows])
    print(f"[data] 书家 {len(cal_ids)} 类, 书体 {len(fnt_ids)} 类, 汉字 {len(hanzi)} 类")

    kw = dict(dim=DIM, steps=a.steps, lr=a.lr, batch=a.batch, k=a.k, temp=a.temp)

    t_cal = supcon_train(feat, cal_lab, len(cal_ids), tag="callig10", **kw)
    np.save(f"{OUT}/callig_table.npy", t_cal)
    json.dump({"classes": cal_ids, "dim": DIM, "source": "supcon_triple"},
              open(f"{OUT}/callig_index.json", "w", encoding="utf-8"),
              ensure_ascii=False)

    t_fnt = supcon_train(feat, fnt_lab, len(fnt_ids), tag="font3", **kw)
    np.save(f"{OUT}/font_table.npy", t_fnt)
    json.dump({"classes": fnt_ids, "dim": DIM, "source": "supcon_triple"},
              open(f"{OUT}/font_index.json", "w", encoding="utf-8"),
              ensure_ascii=False)

    t_chr = supcon_train(feat, chr_lab, len(hanzi), tag="char", **kw)
    np.save(f"{OUT}/char_table.npy", t_chr)
    json.dump({"classes": hanzi, "dim": DIM, "source": "supcon_triple"},
              open(f"{OUT}/char_index.json", "w", encoding="utf-8"),
              ensure_ascii=False)

    remap = {}
    for r in list(rows) + list(_eval_rows):
        remap[str(int(r["character_id"]))] = hz_idx[r["character"]]
    json.dump(remap, open(f"{OUT}/char_remap.json", "w", encoding="utf-8"),
              ensure_ascii=False)
    # callig/font remap: 原始 id -> 表行索引 (builder 用 sorted 顺序, 与 dataset
    # 自身的 callig_map 顺序不保证一致, 故 dataset 侧必须过这两个 remap)
    json.dump({str(c): i for i, c in enumerate(cal_ids)},
              open(f"{OUT}/callig_remap.json", "w", encoding="utf-8"),
              ensure_ascii=False)
    json.dump({str(s): i for i, s in enumerate(fnt_ids)},
              open(f"{OUT}/font_remap.json", "w", encoding="utf-8"),
              ensure_ascii=False)

    # 体检: 每张表的类间余弦 (前 2000 类抽样)
    for tag, t, n_cls in (("callig", t_cal, len(cal_ids)),
                          ("font", t_fnt, len(fnt_ids)),
                          ("char", t_chr, len(hanzi))):
        w = torch.from_numpy(t).float()
        n = min(n_cls, 2000)
        wn = F.normalize(w[:n], dim=-1)
        off = (wn @ wn.T)[~torch.eye(n, dtype=torch.bool)]
        print(f"[check] {tag}: mean|cos|={off.abs().mean():.4f} "
              f"max={off.abs().max():.4f} (n={n})")
    print("[done]")


if __name__ == "__main__":
    main()
