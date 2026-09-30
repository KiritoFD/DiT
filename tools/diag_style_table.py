#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/diag_style_table.py — 解析 top10 风格表 (callig_script_emb_top10.pt) 的结构、分布与性能。"""
import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)

SLOT = {0: "何绍基_楷", 1: "何绍基_行", 2: "何绍基_隶", 3: "文徵明_楷", 4: "文徵明_行",
        5: "文徵明_隶", 6: "柳公权_楷", 7: "柳公权_行", 8: "欧阳询_楷", 9: "欧阳询_行",
        10: "王羲之_楷", 11: "王羲之_行", 12: "米芾_楷", 13: "米芾_行", 14: "苏轼_楷",
        15: "苏轼_行", 16: "褚遂良_楷", 17: "褚遂良_行", 18: "赵孟頫_楷", 19: "赵孟頫_行",
        20: "赵孟頫_隶", 21: "颜真卿_楷", 22: "颜真卿_行"}

for p in ["assets/callig_script_emb_top10.pt", "assets/callig_script_emb_pretrained.pt",
          "assets/callig_script_emb_top10_hier.pt"]:
    if os.path.exists(p):
        print("FOUND:", p, os.path.getsize(p), "bytes")
print()

for p in ["assets/callig_script_emb_top10.pt"]:
    if not os.path.exists(p):
        continue
    d = torch.load(p, map_location="cpu")
    print("=" * 78)
    print("[表结构]", p)
    print("=" * 78)
    if isinstance(d, dict):
        for k, v in d.items():
            if torch.is_tensor(v):
                print("  %-16s tensor%s  %s" % (k, tuple(v.shape), v.dtype))
            else:
                print("  %-16s %s" % (k, v))
        W = d.get("embedding", d.get("weight"))
        ptc = d.get("pair_to_callig")
    else:
        W = d
        ptc = None
        print("  raw tensor", tuple(W.shape))
    W = W.float()
    print()
    print("=" * 78)
    print("[逐槽位分布] per-row: L2范数 / 均值 / 标准差")
    print("=" * 78)
    n = W.shape[0]
    nn_ = W.norm(dim=1)
    mu = W.mean(dim=1)
    sd = W.std(dim=1)
    for i in range(n):
        print("  slot %2d %-10s |w|=%.4f  mean=%+.4f  std=%.4f" % (
            i, SLOT.get(i, "?"), nn_[i], mu[i], sd[i]))
    print()
    print("  全表 |w|: mean=%.4f std=%.4f min=%.4f max=%.4f" % (
        nn_.mean(), nn_.std(), nn_.min(), nn_.max()))
    print("  全局标量 mean=%+.5f std=%.5f" % (W.mean(), W.std()))

    Wn = F.normalize(W, dim=1)
    pc = Wn @ Wn.t()
    off = pc[~torch.eye(n, dtype=torch.bool)]
    print()
    print("=" * 78)
    print("[塌缩体检] pairwise cos (塌缩基线 0.323)")
    print("=" * 78)
    print("  全体 off-diag cos: mean=%+.4f |mean|=%.4f max=%+.4f min=%+.4f" % (
        off.mean(), off.abs().mean(), off.max(), off.min()))

    if ptc is not None:
        ptc = torch.tensor(ptc)
    else:
        ptc = torch.tensor([0,0,0,1,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,8,9,9])
    same = (ptc[:, None] == ptc[None, :])
    eye = torch.eye(n, dtype=torch.bool)
    sib = same & ~eye
    diff = ~same
    print()
    print("  同书家异书体 cos (兄弟对, n=%d): mean=%+.4f" % (sib.sum(), pc[sib].mean()))
    print("  异书家 cos       (负对,   n=%d): mean=%+.4f" % (diff.sum(), pc[diff].mean()))
    print("  --> 分离度 (兄弟 - 异家) = %+.4f  (越负越说明书体被分开)" % (pc[sib].mean() - pc[diff].mean()))

    print()
    print("=" * 78)
    print("[风格能量] 风格分量 (去均值后) 的幅度, 对应 cond_fusion 的 |e_callig|")
    print("=" * 78)
    Wc = W - W.mean(dim=0, keepdim=True)
    print("  |W - colmean| per row mean=%.4f" % Wc.norm(dim=1).mean())
    print("  原始 |w|            per row mean=%.4f" % nn_.mean())
    print("  风格/总量 比 = %.3f" % (Wc.norm(dim=1).mean() / nn_.mean()))
