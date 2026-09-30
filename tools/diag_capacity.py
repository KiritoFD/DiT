#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/diag_capacity.py — 这个任务**需要多大**网络? 风格注入的信息量上限在哪?

A. PCA: Δ = g_gt - g_std 的内在维度
B. 方差分解(按 字 / 字x书家): 风格到底占 Δ 的多大比例
C. 线性上界(岭回归, CPU): 全局线性 / 每书家线性 -> 判断瓶颈是容量还是架构
D. 风格模板的秩上限(结构性分析)
"""
import os
import sys
import glob
import csv
import re
import numpy as np
import torch

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
sys.stdout.reconfigure(encoding="utf-8")
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
OD = 4 * 32 * 32


def load_all(d):
    L, I = [], []
    for sp in sorted(glob.glob(os.path.join(d, 'shard_*.npz'))):
        with np.load(sp) as z:
            L.append(z['latents'].astype(np.float32))
            I.append(z['img_ids'])
    return np.concatenate(L), np.concatenate(I)


def ridge_cpu(Xtr, Ytr, Xva, Yva, lam_frac=1e-3):
    """CPU 岭回归 (float32)。Xtr 含偏置列。"""
    d = Xtr.shape[1]
    A = Xtr.t() @ Xtr + (lam_frac * len(Xtr)) * torch.eye(d)
    W = torch.linalg.solve(A, Xtr.t() @ Ytr)
    return float((Xva @ W - Yva).pow(2).mean())


def main():
    G, gid = load_all('data/top10_style23/shards_std')
    T, tid = load_all('data/top10_style23/shards_aux_skel3')
    assert (gid == tid).all()
    N = G.shape[0]

    rows = list(csv.DictReader(open('assets/train_top10_style23.csv', encoding='utf-8')))
    # 列: image_path, calligrapher, script, character, calligrapher_id, script_id,
    #     character_id, glyph_id, aug, std_path, source, src_image_path, slot_name, pair_id, img_id
    # ★ 「风格」的粒度是 **slot**(书家x书体, 23 个, 与 callig_script_emb_top10.pt 对齐),
    #   不是 calligrapher(10 个)。用错粒度会把书体差异算进"字形"里。
    m2 = {}
    for r in rows:
        try:
            iid = int(r['img_id'])
        except (KeyError, ValueError):
            continue
        m2[iid] = (r.get('character_id', ''), r.get('slot_name', ''))
    chars = sorted({v[0] for v in m2.values() if v[0] != ''})
    cals = sorted({v[1] for v in m2.values() if v[1] != ''})
    ch2i = {c: i for i, c in enumerate(chars)}
    ca2i = {c: i for i, c in enumerate(cals)}
    ch = np.array([ch2i.get(m2.get(int(i), ('', ''))[0], -1) for i in gid])
    ca = np.array([ca2i.get(m2.get(int(i), ('', ''))[1], -1) for i in gid])
    ok = (ch >= 0) & (ca >= 0)
    print(f"[data] {G.shape}  字 {len(chars)} 风格槽位 {len(cals)}  有效 {ok.sum()}/{N}")

    Gt = torch.from_numpy(G[ok])
    Tt = torch.from_numpy(T[ok])
    D = (Tt - Gt).reshape(len(Gt), -1)
    tot_v = float(D.pow(2).mean())
    print(f"[data] raw MSE(g_std, g_gt) = {tot_v:.5f}  (= Δ 的每元素方差)")

    # ── A. PCA ────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("A. 目标场 Δ 的内在维度 (PCA)")
    print("=" * 78)
    idx = torch.randperm(len(D))[:6000]
    Xs = (D[idx] - D[idx].mean(0, keepdim=True)).to(DEV)
    Cov = (Xs.t() @ Xs) / len(Xs)
    ev = torch.linalg.eigvalsh(Cov.double().cpu()).flip(0).clamp_min(0)
    cum = ev.cumsum(0) / ev.sum()
    for p in [0.50, 0.80, 0.90, 0.95, 0.99]:
        k = int((cum < p).sum().item()) + 1
        print(f"   {int(p*100):>3}% 方差 -> {k:>5} 个主成分 ({k/OD*100:.1f}% of 4096)")
    del Cov, ev, cum, Xs
    torch.cuda.empty_cache()

    # ── B. 方差分解 ───────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("B. Δ 的方差分解 —— 风格到底占多大")
    print("=" * 78)
    chn = torch.from_numpy(ch[ok])
    can = torch.from_numpy(ca[ok])
    # 1) 组内 (同字同书家) = 不可约
    key = chn * 1000 + can
    v_in, n_in = 0.0, 0
    for k in key.unique():
        sub = D[key == k]
        v_in += float((sub - sub.mean(0, keepdim=True)).pow(2).sum())
        n_in += sub.numel()
    within = v_in / n_in
    # 2) 同字、跨书家 (组间) = 风格
    v_cross, n_cross = 0.0, 0
    for c in chn.unique():
        sub = D[chn == c]
        if len(sub) < 2:
            continue
        # 每书家一组取均值, 再算这些均值之间的方差 (按样本数加权)
        mus, ws = [], []
        for k in can.unique():
            s2 = sub[can[chn == c] == k]
            if len(s2) >= 1:
                mus.append(s2.mean(0))
                ws.append(len(s2))
        if len(mus) < 2:
            continue
        M = torch.stack(mus)
        w = torch.tensor(ws, dtype=M.dtype)
        gm = (M * w[:, None]).sum(0) / w.sum()
        v_cross += float(((M - gm).pow(2).sum(1) * w).sum())
        n_cross += int(w.sum()) * D.shape[1]
    cross = v_cross / max(n_cross, 1)
    print(f"   总方差 (= MSE)                     {tot_v:.5f}   100.0%")
    print(f"   组内 (同字同槽位, 不可约)           {within:.5f}   {within/tot_v*100:5.1f}%")
    print(f"   同字跨槽位 (★风格信号)              {cross:.5f}   {cross/tot_v*100:5.1f}%")
    print(f"   字与字之间 (字形本身)               {tot_v-within-cross:.5f}   "
          f"{(tot_v-within-cross)/tot_v*100:5.1f}%")
    print("   -> 「闭合率」要打败的基线是 g_std, 而 g_std 已经**精确给出字形**;")
    print("      真正的可改进空间 = 风格信号 + 组内噪声。")

    # ── C. 线性上界 ───────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("C. 线性上界 (岭回归, CPU) —— 瓶颈是容量还是架构?")
    print("=" * 78)
    n = len(Gt)
    nsub = min(n, 20000)
    sel = torch.randperm(n)[:nsub]
    Y = D[sel]
    Xg = Gt.reshape(n, -1)[sel]
    Oh = torch.zeros(nsub, len(cals))
    Oh[torch.arange(nsub), can[sel]] = 1.0
    one = torch.ones(nsub, 1)
    ntr = int(nsub * 0.9)
    tr, va = slice(0, ntr), slice(ntr, nsub)
    Ytr, Yva = Y[tr], Y[va]

    for name, Xf in [('全局线性 [g_std, 1]', torch.cat([Xg, one], 1)),
                     ('+ style onehot', torch.cat([Xg, Oh, one], 1))]:
        d = Xf.shape[1]
        mse = ridge_cpu(Xf[tr].contiguous(), Ytr, Xf[va].contiguous(), Yva)
        print(f"   {name:<26} 参数 {d*OD/1e6:>6.1f}M   留出 MSE = {mse:.5f}")
    # 每书家一套线性 = 完整 style x g_std 交互
    tot, cnt = 0.0, 0
    Xc = torch.cat([Xg, one], 1)
    for k in can[sel].unique():
        ii = torch.nonzero(can[sel] == k).squeeze(1)
        if len(ii) < 40:
            continue
        a, b = ii[:int(len(ii) * 0.9)], ii[int(len(ii) * 0.9):]
        mse = ridge_cpu(Xc[a].contiguous(), Y[a], Xc[b].contiguous(), Y[b])
        tot += mse * len(b) * OD
        cnt += len(b) * OD
    print(f"   每槽位一套线性 (完整交互)     参数 {len(cals)*(OD+1)*OD/1e6:>6.0f}M   "
          f"留出 MSE = {tot/max(cnt,1):.5f}")
    print(f"\n   对照: conv net(width128, 6.44M) 在 probe 空间 MSE = 0.32;")
    print(f"         probe 空间基线 = 0.45693, 输出真值的下界 = 0.01464")

    # ── D. 风格模板的秩 ───────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("D. 风格模板的**秩上限** (结构性缺陷)")
    print("=" * 78)
    for nm, d_in, d_out in [('style_off   (路径2 全分辨率偏移底图)', 128, 2 * 32 * 32),
                            ('stroke_style(路径6 全分辨率残差)', 128, 4 * 32 * 32),
                            ('style_tok_proj(路径7 cross-attn K/V)', 128, 16 * 256)]:
        print(f"   {nm:<40} Linear({d_in} -> {d_out})  秩 <= {d_in}")
    print("   => 23 个书家的风格模板**全部**落在同一个 128 维子空间里。")
    print("      cross-attn 的 16 个 token 也是同一向量的线性展开, 信息量并未增加,")
    print("      只是换了聚合方式。要真加风格容量, 必须用 **per-书家可学习参数表**。")


if __name__ == "__main__":
    main()
