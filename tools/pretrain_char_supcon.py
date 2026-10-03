#!/usr/bin/env python
"""用跨视图 InfoNCE 预训练「字×书体」表 -> 灌 y_char_embedder。 (v2 正确实现)

## v1 的死因 (2026-10-04 实测, 8000 步白跑)
v1 只查表: zz = norm(E(y)), 同 glyph 的 k 个样本查同一行 -> 正对余弦恒为 1
(自己和自己) -> InfoNCE 吸引项梯度恒 0; 只剩类间排斥, 而 4518 行在 512 维球上
随机初始化已近正交 (|cos|=0.035 ≈ 1/√512) -> 排斥也饱和。且 DINO 特征 feat
加载后从未参与 loss。结果: loss 0.8370 -> 0.8377 (7000 步纹丝不动), 表 = 随机。

## v2 修法: 跨视图对比 (CLIP 式, 两塔都有梯度)
  view_A = W[y_i]            查表行 (要训的产物)
  view_B = g(feat_i)         DINO 特征过轻量投影头
  正对 = 同 glyph 的 (A_i, B_j) 全组合 (含自身对), 负 = 异 glyph。
  同字 k 个样本 -> k 个不同 DINO 特征 -> 正对不再是自身 ✓,
  DINO 结构真正灌进表行 ✓。损失对 A/B 双向对称。

用法:
  python tools/pretrain_char_supcon.py --npz assets/dino_feat_top10_g.npz \
      --label-mode glyph --chars-per-script 7026 --num-classes 56831 \
      --dim 512 --k 4 --batch 4096 --steps 8000 --lr 1e-3 \
      --out-prefix assets/char_script_supcon
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.chdir(os.environ.get("DIT_ROOT", "/root/Workspace/xy/DiT"))


def inspect(npz_path):
    z = np.load(npz_path)
    print(f"[inspect] {npz_path}")
    for k in z.files:
        v = z[k]
        print(f"   {k:<12} shape={getattr(v, 'shape', None)} dtype={getattr(v, 'dtype', None)}")
    ok_glyph = ("glyph_ids" in z.files) or (("chars" in z.files) and ("scripts" in z.files))
    print(f"   -> glyph 模式可用: {ok_glyph}")
    return z


def build_labels(z, mode, chars_per_script):
    """glyph_id = script*chars_per_script + char (字×字体, 与 dit.py:1121 一致)。"""
    if "glyph_ids" in z.files:
        g = z["glyph_ids"].astype(np.int64)
        return g if mode == "glyph" else (g % chars_per_script)
    if "chars" not in z.files:
        raise SystemExit(f"[FATAL] npz 无 'glyph_ids' 也无 'chars' (现有 {list(z.files)})")
    chars = z["chars"].astype(np.int64)
    if mode == "char":
        return chars
    if "scripts" not in z.files:
        raise SystemExit("[FATAL] glyph 模式需要 'scripts' 列")
    return z["scripts"].astype(np.int64) * chars_per_script + chars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="assets/dino_feat_top10_g.npz")
    ap.add_argument("--out-prefix", default="assets/char_script_supcon")
    ap.add_argument("--label-mode", choices=["glyph", "char"], default="glyph")
    ap.add_argument("--chars-per-script", type=int, default=7026)
    ap.add_argument("--num-classes", type=int, default=0, help="0=按标签最大值+1")
    ap.add_argument("--dim", type=int, default=512, help="== config 的 char_embed_dim")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=4096,
                    help="每步样本数 = (batch//k) 类 × k 样本。v1 用 16384 纯浪费"
                         "(logits 16384² 白烧显存); 4096 = 1024 类 × 4 足够。")
    ap.add_argument("--temp", type=float, default=0.07)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--k", type=int, default=4, help="每类样本数 (>=2 才有跨样本正对)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inspect", action="store_true")
    a = ap.parse_args()

    if a.inspect:
        inspect(a.npz)
        return 0

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[dev] {dev}")

    z = np.load(a.npz)
    feat_np = z["feat"].astype(np.float32)
    feat_dim = feat_np.shape[1]
    feat = torch.from_numpy(feat_np).to(dev)
    lab = torch.from_numpy(build_labels(z, a.label_mode, a.chars_per_script)).to(dev)
    ok = lab >= 0
    print(f"[supcon] mode={a.label_mode}: {len(lab)} 样本 -> 命中 {int(ok.sum())} "
          f"(丢弃 {int((~ok).sum())}); feat_dim={feat_dim}")
    feat, lab = feat[ok], lab[ok]
    n_cls = a.num_classes if a.num_classes > 0 else int(lab.max().item()) + 1
    uniq = len(set(lab.tolist()))
    print(f"[supcon] 类别数 {n_cls} (唯一出现的 {uniq})")

    # ── 两塔 ──
    W = nn.Embedding(n_cls, a.dim).to(dev)              # 查表塔 (最终产物)
    nn.init.normal_(W.weight, std=0.02)                 # 小 init: 变化可归因于训练
    P = nn.Sequential(nn.Linear(feat_dim, a.dim), nn.GELU(),
                      nn.Linear(a.dim, a.dim)).to(dev)  # 特征塔 (辅助, 只为把 feat 映进表空间)
    opt = torch.optim.Adam(list(W.parameters()) + list(P.parameters()), lr=a.lr)

    # 按类组批: 每类 k 个样本, 保证跨样本正对存在
    _by_cls = defaultdict(list)
    for _i, _g in enumerate(lab.tolist()):
        _by_cls[int(_g)].append(_i)
    _multi = np.array(sorted(g for g, v in _by_cls.items() if len(v) >= 2), dtype=np.int64)
    _C = max(1, a.batch // a.k)
    print(f"[supcon] 可做正样本的类 {len(_multi)} / {len(_by_cls)}; "
          f"组批 k={a.k}, 每批类数={min(_C, len(_multi))}")
    if len(_multi) < 2:
        raise SystemExit("[FATAL] 可做正样本的类不足 2 个")

    _seen = sorted(_by_cls.keys())
    _seen_t = torch.tensor(_seen, device=dev)

    # 类中心 (DINO 特征均值, 投影后): 表行应向它对齐
    _cent_cache = {}
    with torch.no_grad():
        for g in _seen:
            _cent_cache[g] = F.normalize(P(feat[_by_cls[g]]).mean(0), dim=-1)
        _cent_mat = torch.stack([_cent_cache[g] for g in _seen])      # (S, d)
    _cent_ids = torch.tensor(_seen, device=dev)
    # P 在训练, 类中心缓存会过期 -> 每 500 步刷新
    def refresh_centers():
        with torch.no_grad():
            for g in _seen:
                _cent_cache[g] = F.normalize(P(feat[_by_cls[g]]).mean(0), dim=-1)
            _cent_mat.copy_(torch.stack([_cent_cache[g] for g in _seen]))

    def metrics():
        with torch.no_grad():
            w = F.normalize(W.weight[_seen_t], dim=-1)
            off = (w @ w.T)[~torch.eye(len(_seen_t), dtype=torch.bool, device=dev)]
            sub = torch.randperm(len(_seen_t), device=dev)[:1024]
            cm = F.normalize(_cent_mat[sub], dim=-1)
            wm = F.normalize(W.weight[_cent_ids[sub]], dim=-1)
            al = (cm * wm).sum(-1)          # 表行 vs 该类 DINO 中心的余弦 = 真正该涨的量
        return off.abs().mean().item(), off.abs().max().item(), al.mean().item()

    print("[supcon] v2 跨视图 InfoNCE: view_A=表行, view_B=g(DINO feat); "
          "正对=同 glyph 跨样本对 | 对齐度初值应≈0, 终值应显著>0")

    t0 = time.time()
    for step in range(1, a.steps + 1):
        sel = np.random.choice(_multi, size=min(_C, len(_multi)), replace=False)
        idxs = []
        for g in sel:
            v = _by_cls[int(g)]
            idxs += list(v) if len(v) <= a.k else list(np.random.choice(v, size=a.k, replace=False))
        idx = torch.tensor(idxs, device=dev, dtype=torch.long)
        y = lab[idx]
        zA = F.normalize(W(y), dim=-1)                 # (B, d) 表行
        zB = F.normalize(P(feat[idx]), dim=-1)         # (B, d) 投影特征
        logits = zA @ zB.T / a.temp                    # 跨视图 (B,B)
        tgt = (y[:, None] == y[None, :]).float()       # 同 glyph = 正 (含自身对)
        # CLIP 式双向 InfoNCE
        loss_a = -((logits.log_softmax(1) * tgt).sum(1) / tgt.sum(1).clamp_min(1)).mean()
        loss_b = -((logits.log_softmax(0) * tgt).sum(0) / tgt.sum(0).clamp_min(1)).mean()
        loss = 0.5 * (loss_a + loss_b)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 1 or step % 500 == 0 or step == a.steps:
            mc, mx, al = metrics()
            print(f"  step {step}: loss={loss.item():.4f} 对齐度={al:.4f} "
                  f"|cos|={mc:.4f} max={mx:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if step % 500 == 0:
            refresh_centers()

    w = W.weight.detach().cpu().numpy()
    print(f"\n[最终] 表 {w.shape} (训到 {len(_seen)} 行)")
    np.save(f"{a.out_prefix}_table.npy", w)
    glyphs = sorted({int(v) for v in lab.tolist()})
    json.dump({"glyphs": glyphs, "label_mode": a.label_mode,
               "chars_per_script": a.chars_per_script, "dim": a.dim,
               "n_classes": n_cls, "source": "supcon_char_script_v2", "npz": a.npz},
              open(f"{a.out_prefix}_index.json", "w", encoding="utf-8"), ensure_ascii=False)
    print(f"[ok] -> {a.out_prefix}_table.npy / _index.json")
    print("\nconfig 里这样接:")
    print(f'  "char_dino_embeddings": "{a.out_prefix}_table.npy",')
    print(f'  "char_dino_index": "{a.out_prefix}_index.json",')
    print(f'  "char_embed_dim": {a.dim},')
    return 0


if __name__ == "__main__":
    sys.exit(main())
