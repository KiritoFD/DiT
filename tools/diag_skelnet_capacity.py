#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/diag_skelnet_capacity.py — 6.44M 到底够不够? 用「小样本记忆测试」定性。

逻辑:
  - 取固定 1024 条样本, 关掉全部正则/对比/增强, 只留 probe L2 + w_mass。
  - 训 width 128 (6.44M) 与 width 256 (21.8M), 看**训练集** MSE 能压到多低。
  - 若 width128 就能压到接近下界(0.0146), 说明**容量不是瓶颈**, 问题在优化/归纳偏置/正则。
  - 若两者都卡在 ~0.3, 说明容量确实不够。
"""
import os
import sys
import glob
import numpy as np
import torch
import torch.nn.functional as F

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
sys.stdout.reconfigure(encoding="utf-8")
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SUBSET = 1024
STEPS = 3000
BATCH = 128


def load_all(d):
    L, I = [], []
    for sp in sorted(glob.glob(os.path.join(d, 'shard_*.npz'))):
        with np.load(sp) as z:
            L.append(z['latents'].astype(np.float32))
            I.append(z['img_ids'])
    return np.concatenate(L), np.concatenate(I)


def main():
    from src.model.deform_skel import DeformSkel
    from src.train.latent_structure import LatentSkelProbe

    G, gid = load_all('data/top10_style23/shards_std')
    T, tid = load_all('data/top10_style23/shards_aux_skel3')
    Y = torch.load('assets/callig_script_emb_top10.pt', map_location='cpu', weights_only=False)
    if isinstance(Y, dict):
        Y = Y.get('emb', Y.get('weight', list(Y.values())[0]))
    Y = Y.float()
    print(f"[data] G{G.shape} T{T.shape} style{tuple(Y.shape)}")

    rng = np.random.RandomState(0)
    sel = rng.choice(len(G), SUBSET, replace=False)
    Gs = torch.from_numpy(G[sel]).to(DEV)
    Ts = torch.from_numpy(T[sel]).to(DEV)
    Ys = Y.to(DEV)

    # 每条的风格: 用 train csv 的 slot
    import csv, re
    rows = list(csv.DictReader(open('assets/train_top10_style23.csv', encoding='utf-8')))
    id2slot = {}
    slots = sorted({r['slot_name'] for r in rows})
    s2i = {s: i for i, s in enumerate(slots)}
    for r in rows:
        try:
            id2slot[int(r['img_id'])] = s2i[r['slot_name']]
        except Exception:
            pass
    sid = torch.tensor([id2slot.get(int(i), 0) for i in np.array(gid)[sel]], device=DEV)
    print(f"[data] 子集 {SUBSET} 条, 覆盖 {len(set(sid.tolist()))} 个槽位")

    ck = torch.load('assets/structure_probes/latent_skel_probe_v1_mse/best.pt',
                    map_location='cpu', weights_only=False)
    pa = ck['args']
    probe = LatentSkelProbe(pa['in_channels'], pa['out_channels'], pa['width'], pa['depth'])
    probe.load_state_dict(ck['model'], strict=False)
    probe = probe.to(DEV).eval()
    for p in probe.parameters():
        p.requires_grad_(False)

    with torch.no_grad():
        base = float(F.mse_loss(probe(Gs).float(), Ts))
        floor = float(F.mse_loss(probe(Ts).float(), Ts))
    print(f"\n[ref] probe 空间: 基线 MSE(probe(g_std),g_gt) = {base:.5f} | "
          f"下界 MSE(probe(g_gt),g_gt) = {floor:.5f}")

    print(f"\n{'width':>7}{'params':>12}{'step0':>10}{'step1000':>11}{'step3000':>11}"
          f"{'闭合':>8}{'吃下可达':>10}")
    print("  " + "-" * 68)
    for w in [128, 256]:
        torch.manual_seed(0)
        m = DeformSkel(cond_dim=128, ch=4, grid=32, style_ch=32, width=w, max_off=6.0,
                       coarse=8, residual=0, res_cap=1.0, blur=0, dt_ch=1, affine=1, ckpt=0,
                       stroke_mod=1, stroke_cap=1.0, gate_radius=0.25, topo_mode=1,
                       warp_iters=1, film_mode='film', style_tokens=16, attn_heads=4,
                       preserve_amp=1).to(DEV)
        np_ = sum(p.numel() for p in m.parameters())
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=1e-4)
        rec = {}
        for st in range(STEPS + 1):
            if st in (0, 1000, 3000):
                m.eval()
                with torch.no_grad():
                    g2 = m(Gs, Ys[sid])
                    rec[st] = float(F.mse_loss(probe(g2).float(), Ts))
                m.train()
            if st == STEPS:
                break
            bi = torch.randint(0, SUBSET, (BATCH,), device=DEV)
            g2 = m(Gs[bi], Ys[sid[bi]])
            loss = F.mse_loss(probe(g2).float(), Ts[bi])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sch.step()
        closed = 1 - rec[3000] / base
        got = (base - rec[3000]) / max(base - floor, 1e-9)
        print(f"{w:>7}{np_:>12,}{rec[0]:>10.4f}{rec[1000]:>11.4f}{rec[3000]:>11.4f}"
              f"{closed*100:>7.1f}%{got*100:>9.1f}%")
        del m, opt
        torch.cuda.empty_cache()

    print("\n判读: 若 width128 的 step3000 已经接近下界 -> 容量不是瓶颈;")
    print("      若两个宽度都卡在 ~0.3 -> 容量/优化确实不够。")


if __name__ == "__main__":
    main()
