#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""build_triple_tables_raw.py — 在 raw 大语料上, 用**仓库既有的 v2 跨视图 SupCon** 独立训三张表。

不新造方法, 直接用已验证的资产:
  - tools/build_triple_tables.py::supcon_train  (三张表**各自独立**训练, CLIP 式两塔)
  - tools/pretrain_char_supcon.py 的 v1 死因与 v2 修法结论
  - tools/pretrain_multistyle_supcon.py 的实测结论

三条来自仓库实测的硬约束 (照做, 不要自由发挥):
  1) **表行初始化必须是小随机** (std=0.02), 不能用 QR 正交。
     QR 正交使行间余弦恰为 0 -> 所有 logits 恰为 0 -> softmax 恰好均匀 ->
     对称鞍点, 梯度信号被抵消 -> loss 卡在 ln(B) 一动不动 (本次踩到过)。
  2) **不要向 DINO 质心锚定** (w_anchor=0)。实测 DINO 的"书家"pair 质心余弦
     = 0.956, 原生 DINO 表行间余弦 0.860 -> 几乎共线; 锚定 = 把表往"不可分"拉。
  3) **batch 用 4096, 不要用 16384** (pretrain_char_supcon 注释: 16384² logits 纯浪费)。
     batch = (1024 类) x (k=4 样本), k=4 同时满足"每类 >=4 才参与训练"。

特征源 (--npz): 同一套代码吃两种特征, 这就是"dino 与 raw rgb 各训一次"的实现
  assets/dino_feat_raw.npz   (冻结 DINOv2 CLS, 393486x384, 已存在)
  assets/rgb_feat_raw.npz    (从零 RGB 编码器抽出, 由 tools/extract_rgb_feats_raw.py 产出)

用法:
  PYTHONPATH=. python tools/build_triple_tables_raw.py --npz assets/dino_feat_raw.npz \
      --out assets/triple_tables_big_dino
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir(os.environ.get("DIT_ROOT", "/root/Workspace/xy/DiT"))

CSV_RAW = "/root/Workspace/xy/UNIFIED_RAW/meta/train_clean.csv"
DEV = "cuda"


def supcon_train(feat, lab, n_cls, dim, *, tag, steps, lr, batch, k, temp,
                 holdout_idx=None, log_every=500):
    """v2 跨视图 SupCon: view_A = 表行 W(y), view_B = P(feat); 正对 = 同类跨样本 (含自身对)。

    表行**只靠标签驱动**, 不锚定任何外部几何 (见文件头约束 2)。
    """
    dev = feat.device
    lab_t = torch.from_numpy(lab.astype(np.int64)).to(dev)

    W = nn.Embedding(n_cls, dim).to(dev)
    nn.init.normal_(W.weight, std=0.02)          # ★ 约束 1: 小随机, 不用 QR
    P = nn.Sequential(nn.Linear(feat.size(1), dim), nn.GELU(),
                      nn.Linear(dim, dim)).to(dev)
    opt = torch.optim.Adam(list(W.parameters()) + list(P.parameters()), lr=lr)

    by = defaultdict(list)
    for i, g in enumerate(lab.tolist()):
        by[int(g)].append(i)
    elig = np.array(sorted(g for g, v in by.items() if len(v) >= k), dtype=np.int64)
    C = max(1, batch // k)
    print(f"\n[{tag}] 类数 {n_cls} (出现 {len(by)}), 每类>={k} 可训练 {len(elig)}, "
          f"dim={dim}, 每步 {min(C, len(elig))} 类 x {k} = "
          f"{min(C, len(elig)) * k} 样本", flush=True)
    if len(elig) < 2:
        raise SystemExit(f"[FATAL] {tag}: 可做正对的类不足")

    # 诊断量 (仓库 v2 的原装指标): 表行 vs 该类**投影后质心**的余弦 —— 必须同空间比。
    # 质心只用训练样本 (排除 holdout), 每类最多 8 张, 每次 eval 重算 (P 在动)。
    ho_set = set(holdout_idx.tolist()) if holdout_idx is not None else set()
    _ei, _ec = [], []
    for g in elig.tolist():
        v = [i for i in by[int(g)] if i not in ho_set]
        if not v:
            continue
        pick = v if len(v) <= 8 else list(np.random.choice(v, size=8, replace=False))
        _ei += pick
        _ec += [int(g)] * len(pick)
    ei = torch.tensor(_ei, device=dev, dtype=torch.long)
    ec = torch.tensor(_ec, device=dev, dtype=torch.long)
    print(f"[{tag}] 诊断子集: {len(_ei)} 样本 / {len(set(_ec))} 类 (每类<=8)", flush=True)

    def eval_all():
        with torch.no_grad():
            Wn = F.normalize(W.weight, dim=-1)
            zp = F.normalize(P(feat[ei]), dim=-1)
            csum = torch.zeros(n_cls, dim, device=dev)
            cn = torch.zeros(n_cls, 1, device=dev)
            csum.index_add_(0, ec, zp)
            cn.index_add_(0, ec, torch.ones(ec.size(0), 1, device=dev))
            ctr = F.normalize(csum / cn.clamp_min(1), dim=-1)
            e = torch.from_numpy(elig).to(dev)
            align = (Wn[e] * ctr[e]).sum(-1).mean().item()
            cos = Wn[e] @ Wn[e].t()
            off = ~torch.eye(len(e), dtype=torch.bool, device=dev)
            acc = None
            if holdout_idx is not None and len(holdout_idx):
                hi = torch.from_numpy(holdout_idx).to(dev)
                z = F.normalize(P(feat[hi]), dim=-1)
                pred = (z @ Wn[e].t()).argmax(1)
                acc = (e[pred] == lab_t[hi]).float().mean().item()
        return dict(align=align, mean_abs_cos=cos[off].abs().mean().item(),
                    max_cos=cos[off].max().item(), holdout_top1=acc,
                    n_trained=len(e))

    t0 = time.time()
    best = (-1.0, None, None)
    for step in range(1, steps + 1):
        sel = np.random.choice(elig, size=min(C, len(elig)), replace=False)
        idxs = []
        for g in sel:
            v = by[int(g)]
            idxs += list(v) if len(v) <= k else list(np.random.choice(v, size=k, replace=False))
        idx = torch.tensor(idxs, device=dev, dtype=torch.long)
        y = lab_t[idx]
        zA = F.normalize(W(y), dim=-1)
        zB = F.normalize(P(feat[idx]), dim=-1)
        logits = zA @ zB.t() / temp
        tgt = (y[:, None] == y[None, :]).float()
        loss = 0.5 * (-((logits.log_softmax(1) * tgt).sum(1) / tgt.sum(1).clamp_min(1)).mean()
                      - ((logits.log_softmax(0) * tgt).sum(0) / tgt.sum(0).clamp_min(1)).mean())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step == 1 or step % log_every == 0 or step == steps:
            m = eval_all()
            star = ""
            if m["align"] > best[0]:
                best = (m["align"], W.weight.detach().clone(),
                        F.normalize(W.weight, dim=-1).detach().clone())
                star = " ★"
            print(f"  [{tag} {step:5d}/{steps}] loss {loss.item():.4f} "
                  f"(chance {np.log(zA.size(0)):.3f}) | 对齐度 {m['align']:+.4f}{star} | "
                  f"行|cos| {m['mean_abs_cos']:.4f} max {m['max_cos']:.4f}"
                  + (f" | holdout top-1 {m['holdout_top1']:.4f}"
                     if m["holdout_top1"] is not None else "")
                  + f" ({time.time()-t0:.0f}s)", flush=True)

    if best[1] is None:
        best = (0.0, W.weight.detach().clone(), F.normalize(W.weight, dim=-1).detach())
    return (best[2].cpu().numpy().astype(np.float32), best[0], elig, P, m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="assets/dino_feat_raw.npz")
    ap.add_argument("--out", default="")
    ap.add_argument("--csv", default=CSV_RAW)
    ap.add_argument("--dim-script", type=int, default=32)
    ap.add_argument("--dim-callig", type=int, default=128)
    ap.add_argument("--dim-char", type=int, default=384)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=4096, help="= (batch//k) 类 x k 样本")
    ap.add_argument("--k", type=int, default=4, help="每类样本数; 也是参与训练的最少样本数")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temp", type=float, default=0.07)
    ap.add_argument("--holdout", type=float, default=0.05)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if not a.out:
        a.out = "assets/triple_tables_big_" + os.path.basename(a.npz).replace("feat_", "").replace(".npz", "")
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)

    t_load = time.time()
    z = np.load(a.npz)
    need = ("feat", "scripts", "calligs", "glyph_ids")
    miss = [k for k in need if k not in z.files]
    if miss:
        raise SystemExit(f"[FATAL] npz 缺 {miss}; 现有 {list(z.files)}")
    feat_np = z["feat"].astype(np.float32)
    s_lab, g_lab, c_lab = z["scripts"], z["calligs"], z["glyph_ids"]
    if a.limit:
        feat_np = feat_np[:a.limit]
        s_lab, g_lab, c_lab = s_lab[:a.limit], g_lab[:a.limit], c_lab[:a.limit]
    n_scr = int(s_lab.max()) + 1
    n_cal = int(g_lab.max()) + 1
    n_chr = int(c_lab.max()) + 1
    print(f"[npz] {a.npz} feat={feat_np.shape} 载入 {time.time()-t_load:.1f}s | "
          f"书体 {n_scr} / 书家 {n_cal} / 汉字 {n_chr}", flush=True)
    feat = torch.from_numpy(feat_np).to(DEV)

    # 名字 (只用来写 index.json)
    names_s, names_g, names_c = {}, {}, {}
    for r in csv.DictReader(open(a.csv, encoding="utf-8")):
        names_s[int(r["script_id"])] = r["script"]
        names_g[int(r["calligrapher_id"])] = r["calligrapher"]
        names_c[int(r["character_id"])] = r["character"]
    print(f"[csv] 名字表: 书体 {len(names_s)} / 书家 {len(names_g)} / 汉字 {len(names_c)}",
          flush=True)

    rng = np.random.RandomState(1234)
    hold = np.sort(rng.permutation(len(feat))[:int(len(feat) * a.holdout)])
    print(f"[holdout] {len(hold):,} 张 ({a.holdout:.0%})", flush=True)

    kw = dict(steps=a.steps, lr=a.lr, batch=a.batch, k=a.k, temp=a.temp,
              holdout_idx=hold, log_every=a.log_every)
    out_tabs = {}
    for tag, lab, n_cls, dim, names in (
            ("font", s_lab, n_scr, a.dim_script, names_s),
            ("callig", g_lab, n_cal, a.dim_callig, names_g),
            ("char", c_lab, n_chr, a.dim_char, names_c)):
        Wn, align, elig, P, _ = supcon_train(feat, lab, n_cls, dim, tag=tag, **kw)
        # 稀有类(K<k)兜底: 该类质心过**训好的**投影 P, 保证下游任何 id 都查得到
        with torch.no_grad():
            full = torch.from_numpy(Wn).to(DEV)
            has = np.array([int((lab == c).sum()) for c in range(n_cls)]) > 0
            rare = np.array([c for c in range(n_cls)
                             if has[c] and c not in set(elig.tolist())])
            if len(rare):
                rt = torch.from_numpy(rare).to(DEV)
                cm = torch.zeros(n_cls, feat.size(1), device=DEV)
                cn = torch.zeros(n_cls, 1, device=DEV)
                lt = torch.from_numpy(lab.astype(np.int64)).to(DEV)
                cm.index_add_(0, lt, feat)
                cn.index_add_(0, lt, torch.ones(lt.size(0), 1, device=DEV))
                full[rt] = F.normalize(P(cm[rt] / cn[rt].clamp_min(1)), dim=-1)
            Wn = full.cpu().numpy()
        print(f"[{tag}] 稀有类兜底 {len(rare)} 类; 无样本类 {int((~has).sum())} 类 "
              f"(留随机行)", flush=True)
        np.save(f"{a.out}/{tag}_table.npy", Wn)
        json.dump({"classes": [names.get(i, "") for i in range(n_cls)], "dim": int(dim),
                   "source": f"supcon_v2@{os.path.basename(a.npz)}"},
                  open(f"{a.out}/{tag}_index.json", "w", encoding="utf-8"),
                  ensure_ascii=False)
        json.dump({str(i): i for i in range(n_cls)},
                  open(f"{a.out}/{tag}_identity_remap.json", "w", encoding="utf-8"),
                  ensure_ascii=False)
        out_tabs[tag] = (Wn, align, len(elig))
        print(f"[save] {a.out}/{tag}_table.npy {Wn.shape}  最佳对齐度 {align:+.4f}", flush=True)

    json.dump({"npz": a.npz, "args": vars(a), "npz_keys": list(z.files),
               "n_rows": {k: int(v[0].shape[0]) for k, v in out_tabs.items()},
               "best_align": {k: float(v[1]) for k, v in out_tabs.items()},
               "n_trained": {k: int(v[2]) for k, v in out_tabs.items()}},
              open(f"{a.out}/train_log.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n[done] {a.out}", flush=True)


if __name__ == "__main__":
    main()
