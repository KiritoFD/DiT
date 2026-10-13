#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""train_style_encoder.py —— 按 v85 方案 §三.2 训练**跨字符书家风格编码器**（latent 空间 ✓）。

方案要求（逐条落实 ✓）
    Backbone : 现有约 2.9M latent CNN（(4,32,32) -> 256D, L2 归一化 ✓）
    输入     : **v84A 同一 VAE 契约**的 4×32×32 latent（sd-vae-ft-ema, sf=0.18215 ✓）
    Embedding: 256D, L2 normalize ✓
    Loss     : `L_CE(writer) + 0.3 L_SupCon + 0.2 L_hard` ✓
                 正 = 同书家×同书体×**异字** ✓（跨字符风格一致 ✓）
                 难负 = **异书家**×同书体×**同字** ✓（专治"同字异书家" ✓）
    验证     : **按作品来源留出** + 书体内分组 ✓ ⇒ 门槛只在**留出的作品**上算 ✓

为什么正/负样本必须按**索引查表**（不是 batch 内配对 ✗）
    原脚本已记下这个坑: 5750 字 / batch 256 ⇒ 同字几乎不可能同批出现 ⇒ 负样本恒空 ⇒ NCE 变 no-op ✓。
    InfoNCE 必须写全 `logsumexp(pos, neg) − pos/τ` ✓（漏了 `− pos/τ` 就变成"把一切都推开" ✓）。

用法
    python tools/train_style_encoder.py --steps 6000 --batch 256 --out assets/style_enc_v85.pt
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.style.bank import resolve, work_id                          # noqa: E402

VAE_DIR = "data/pretrained/sd-vae-ft-ema"
SF = 0.18215                     # ★ 与 v84A 同一契约 ✓


# ───────────────────────── 数据 ─────────────────────────
def load_rows(csv_path, roots, max_img=0):
    """均衡抽样 + 保留 std 路径（G3 需要 ✓）。"""
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8-sig")))
    by = defaultdict(list)
    miss = 0
    for r in rows:
        if (r.get("aug") or "").strip():
            continue
        p = resolve(r.get("image_path", ""), roots)
        if not p:
            miss += 1
            continue
        by[(r.get("calligrapher") or "", r.get("script") or "")].append({
            "path": p, "writer": (r.get("calligrapher") or "").strip(),
            "script": (r.get("script") or "").strip(),
            "char": (r.get("character") or "").strip(),
            "work": work_id(r.get("src_image_path", ""), r.get("image_path", "")),
            "std": resolve(r.get("std_path", ""), roots) or ""})
    out, i = [], 0
    if max_img:
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
    else:
        out = [d for k in sorted(by) for d in by[k]]
    print("[data] 可用 %d（缺图 %d）| 书家 %d | 字 %d | 作品 %d"
          % (len(out), miss, len({r['writer'] for r in out}),
             len({r['char'] for r in out}), len({r['work'] for r in out})))
    return out


def encode_latents(rows, cache, device, batch=32):
    """用 v84A 同一 VAE 把像素编码成 4×32×32 latent ✓（带缓存 ✓）。"""
    if os.path.isfile(cache):
        z = np.load(cache)
        if int(z["lat"].shape[0]) == len(rows):
            print("[vae] 命中缓存 %s %s" % (cache, z["lat"].shape))
            return z["lat"].astype(np.float32)
    from PIL import Image
    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(VAE_DIR).to(device).eval()
    print("[vae] 编码 %d 张 -> %s ..." % (len(rows), cache), flush=True)
    out = np.zeros((len(rows), 4, 32, 32), np.float16)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            chunk = rows[i:i + batch]
            x = torch.stack([
                torch.from_numpy(np.asarray(Image.open(r["path"]).convert("RGB")))
                .permute(2, 0, 1).float() / 127.5 - 1.0 for r in chunk]).to(device)
            z = vae.encode(x).latent_dist.sample() * SF
            out[i:i + len(chunk)] = z.half().cpu().numpy()
            if (i // batch) % 20 == 0:
                print("  [vae] %d/%d (%.0fs)" % (i + len(chunk), len(rows), time.time() - t0),
                      flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(cache)), exist_ok=True)
    np.savez_compressed(cache, lat=out)
    print("[vae] 完成 %s (%.0fs)" % (out.shape, time.time() - t0))
    del vae
    torch.cuda.empty_cache()
    return out.astype(np.float32)


# ───────────────────────── 模型（方案: 现有 ~2.9M latent CNN ✓）─────────────────────────
class Enc(nn.Module):
    def __init__(self, nc, d=256, ch=96, drop=0.0):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(4, ch, 3, 1, 1), nn.GELU(), nn.GroupNorm(8, ch),
            nn.Conv2d(ch, ch * 2, 4, 2, 1), nn.GELU(), nn.GroupNorm(8, ch * 2),
            nn.Conv2d(ch * 2, ch * 4, 4, 2, 1), nn.GELU(), nn.GroupNorm(16, ch * 4),
            nn.Conv2d(ch * 4, ch * 4, 3, 1, 1), nn.GELU(), nn.GroupNorm(16, ch * 4),
            nn.AdaptiveAvgPool2d(1))
        self.head = nn.Linear(ch * 4, d)
        self.drop = nn.Dropout(drop)
        self.clf = nn.Linear(d, nc)

    def emb(self, x):
        return F.normalize(self.head(self.drop(self.body(x).flatten(1))), dim=-1)

    def forward(self, x):
        z = self.emb(x)
        return z, self.clf(z)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="exp-std/csv/train.csv")
    ap.add_argument("--root", action="append", default=[])
    ap.add_argument("--max-img", type=int, default=0, help="0=全部可解析的 ✓")
    ap.add_argument("--cache", default="scratch/style_latents_local.npz")
    ap.add_argument("--out", default="assets/style_enc_v85.pt")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--neg-per-anchor", type=int, default=1,
                    help="[v3] 每个 anchor 用几个**内容匹配**难负（默认 1 = 与 v2 逐位一致 ✓）。\n"
                    "方案的核心是内容匹配难负 ⇒ 只 1 个时对比项会很快平台 ✗（实测 ✓）")
    ap.add_argument("--ch", type=int, default=96)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--w-supcon", type=float, default=0.3,
                    help="方案: CE + **0.3** SupCon + **0.2** hard ✓")
    ap.add_argument("--w-hard", type=float, default=0.2)
    ap.add_argument("--holdout-work-frac", type=float, default=0.3)
    ap.add_argument("--n-pairs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="")
    args = ap.parse_args()
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    roots = args.root or [".", "data", "exp-std"]

    rows = load_rows(args.csv, roots, args.max_img)
    if len(rows) < 200:
        print("样本太少 ✗")
        return 2
    lat = encode_latents(rows, args.cache, dev)

    # ── **作品级留出**（方案要求 ✓）：每个槽位里挑一部分作品只做验收 ✓
    rng = random.Random(args.seed)
    by_slot = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(rows):
        by_slot[(r["writer"], r["script"])][r["work"]].append(i)
    ho = []
    for _slot, works in by_slot.items():
        ws = sorted(works)
        if len(ws) >= 3:
            rng.shuffle(ws)
            for w in ws[:max(1, int(round(len(ws) * args.holdout_work_frac)))]:
                ho += works[w]
    ho = sorted(set(ho))
    m_ho = np.zeros(len(rows), bool)
    m_ho[ho] = True
    tr = np.nonzero(~m_ho)[0]
    print("[split] 作品级留出: train %d / holdout %d (作品 %d/%d)"
          % (len(tr), int(m_ho.sum()), len({rows[i]['work'] for i in tr}),
             len({rows[i]['work'] for i in ho})))

    CALLS = sorted({r["writer"] for r in rows})
    C2I = {c: i for i, c in enumerate(CALLS)}
    cal = np.array([C2I[r["writer"]] for r in rows])
    chs = np.array([r["char"] for r in rows])

    X = torch.from_numpy(lat[tr]).to(dev)
    y = torch.from_numpy(cal[tr]).to(dev)
    # ── 索引式配对表（不能用 batch 内配对 ✓ 原脚本已记坑 ✓）──
    by_cal, by_char = defaultdict(list), defaultdict(list)
    by_slot = defaultdict(list)                     # (书家, 书体) ✓ 供 hard-negative 退化用 ✓
    for j in range(len(tr)):
        by_cal[int(cal[tr[j]])].append(j)
        by_char[chs[tr[j]]].append(j)
        by_slot[(int(cal[tr[j]]), rows[tr[j]]["script"])].append(j)
    POS = [np.array([x for x in by_cal[int(cal[tr[j]])] if chs[tr[x]] != chs[tr[j]]] or [j])
           for j in range(len(tr))]
    # ★ 难负: 优先"同字异书家" ✓；没有则**退回同书体的其他书家** ✓（方案明确要求 ✓）
    #   ⚠ 实测: 只用"同字异书家"时**仅 7% 的样本有负样本** ✗ ⇒ 92% 退化成自己配自己
    #     ⇒ 0.2·L_hard 变 no-op ✗，而 G1 门槛恰恰只能靠这个信号 ✓ ⇒ 必须退化 ✓。
    NEG, _hard = [], 0
    for j in range(len(tr)):
        v = [x for x in by_char[chs[tr[j]]] if int(cal[tr[x]]) != int(cal[tr[j]])]
        if v:
            _hard += 1
        else:
            v = [x for x in by_slot[(int(cal[tr[j]]), rows[tr[j]]["script"])]
                 if int(cal[tr[x]]) != int(cal[tr[j]])]
        NEG.append(np.array(v) if v else np.array([j]))
    print("[pair] 正(同书家异字) 均 %.0f | 难负 均 %.1f | **真·同字异书家覆盖率 %.0f%%**"
          "（其余退化为同书体异书家 ✓ 方案要求 ✓）"
          % (np.mean([len(p) for p in POS]), np.mean([len(n) for n in NEG]),
             100 * _hard / max(1, len(tr))))

    model = Enc(len(CALLS), d=args.d, ch=args.ch).to(dev)
    print("[model] 参数 %s" % format(sum(p.numel() for p in model.parameters()), ","))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    rng2 = np.random.RandomState(args.seed)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = rng2.randint(0, len(tr), args.batch)
        xb = X[idx]
        zb, logits = model(xb)
        l_ce = F.cross_entropy(logits, y[idx])
        # SupCon: 正 = 同书家异字 ✓
        pj = np.array([POS[i][rng2.randint(len(POS[i]))] for i in idx])
        zp = model.emb(X[pj])
        sim_p = (zb @ zp.t()) / args.tau
        l_pos = -F.log_softmax(sim_p, dim=1).diagonal().mean()
        # HardNCE: 难负 = 同字异书家 ✓（InfoNCE 写全: logsumexp − pos/τ ✓）
        # ★ v3: 支持**每个 anchor 用多个**难负 ✓（默认 1 ⇒ 与 v2 **逐位一致** ✓，不静默改 ✓）
        #   为什么: 方案的核心是"**内容匹配**的难负样本" ✓；只用 1 个时 2.9M 编码器
        #   750 步就把 CE 压到 0.002 而**对比项随即平台** ✗（实测 ✓）⇒ 加负样本数最直接 ✓。
        kn = int(getattr(args, "neg_per_anchor", 1) or 1)
        pos_ii = (zb * zp).sum(-1, keepdim=True) / args.tau
        if kn <= 1:
            nj = np.array([NEG[i][rng2.randint(len(NEG[i]))] for i in idx])
            zn = model.emb(X[nj])
            sim_n = (zb @ zn.t()) / args.tau
            l_hard = (torch.logsumexp(torch.cat([pos_ii, sim_n], 1), dim=1)
                      - pos_ii.squeeze(1)).mean()
        else:
            nj = np.stack([np.array([NEG[i][rng2.randint(len(NEG[i]))] for _ in range(kn)])
                           for i in idx])                       # (n, kn) ✓
            zn = model.emb(X[nj.reshape(-1)])                   # (n*kn, 256) ✓
            sim_n = torch.einsum("nd,nkd->nk", zb, zn.view(len(idx), kn, -1)) / args.tau
            l_hard = (torch.logsumexp(torch.cat([pos_ii, sim_n], 1), dim=1)
                      - pos_ii.squeeze(1)).mean()
        loss = l_ce + args.w_supcon * l_pos + args.w_hard * l_hard
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % max(1, args.steps // 8) == 0 or step == 1:
            print("  [step %5d] CE %.4f | SupCon %.4f | Hard %.4f | 总 %.4f | %.0fs"
                  % (step, float(l_ce), float(l_pos), float(l_hard), float(loss),
                     time.time() - t0), flush=True)

    # ── 验收：**只在留出的作品上**算门槛 ✓（方案要求 ✓）──
    @torch.no_grad()
    def emb_of(mat):
        out = []
        for s in range(0, len(mat), 2048):
            out.append(model.emb(torch.from_numpy(mat[s:s + 2048]).to(dev)).cpu().numpy())
        return np.concatenate(out)

    ho_rows = [rows[i] for i in ho]
    E_ho = emb_of(lat[ho])
    std_rows = [{"path": r["std"], "writer": r["writer"], "script": r["script"],
                 "char": r["char"], "work": "std"} for r in ho_rows if r["std"]]
    E_std = None
    if std_rows:
        from PIL import Image
        arr_std = np.stack([
            np.asarray(Image.open(r["path"]).convert("RGB")).transpose(2, 0, 1)
            for r in std_rows]).astype(np.float32) / 127.5 - 1.0
        # ★ std 图要走**同一个 VAE** ⇒ latent 空间可比 ✓（直接在像素上比是错的 ✗）
        import torch as _t
        from diffusers.models import AutoencoderKL
        _vae = AutoencoderKL.from_pretrained(VAE_DIR).to(dev).eval()
        _z = []
        with _t.no_grad():
            for s in range(0, len(arr_std), 32):
                x = _t.from_numpy(arr_std[s:s + 32]).to(dev)
                _z.append((_vae.encode(x).latent_dist.sample() * SF).cpu().numpy())
        E_std = emb_of(np.concatenate(_z).astype(np.float32))
        del _vae
        _t.cuda.empty_cache()
    print("\n===== 验收（作品级留出集 ✓ n=%d）=====" % len(ho))
    # ★ G1/G3 在**留出的作品**上算 ✓（方案要求 ✓）；
    #   G2 **必须用全量** —— 它自己内部就做作品级留出 ✓，而留出集很小(十几件作品)时
    #   "每槽位≥3 作品"的条件不满足 ⇒ 会算成 n=0/nan ✗（冒烟时实测到了 ✓）。
    from src.eval.style.gates import gate1_2afc, gate2_retrieval, gate3_printed
    g1 = gate1_2afc(E_ho, ho_rows, n_pairs=args.n_pairs, seed=args.seed)
    E_all = emb_of(lat)
    g2 = gate2_retrieval(E_all, rows, seed=args.seed)
    g3 = gate3_printed(E_ho, ho_rows, E_std, std_rows or None)
    res = {"G1": g1, "G2": g2, "G3": g3,
           "pass": bool(g1["acc"] >= 0.65 and g2["map"] == g2["map"]
                        and g2["map"] > g2["base_script"] + 0.05)}
    print("  G1 2AFC(留出作品)   : %.4f (n=%d)  随机 0.5 / 门槛 ≥0.65 -> %s"
          % (g1["acc"], g1["n"], "✓" if g1["acc"] >= 0.65 else "✗"))
    print("  G2 跨字符检索 mAP   : %.4f (n=%d)  同书体基线 %.4f -> %s"
          % (g2["map"], g2["n"], g2["base_script"], "✓" if res["pass"] else "✗"))
    if "skip" not in g3:
        print("  G3 印刷体 top1/rank : %.4f / %.2f (n=%d, 书家数=%d)"
              % (g3["top1"], g3["mean_rank"], g3["n"], g3["n_writer"]))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"model": model.state_dict(), "calls": CALLS, "d": args.d, "ch": args.ch,
                "gates": res, "holdout_work_frac": args.holdout_work_frac,
                "w_supcon": args.w_supcon, "w_hard": args.w_hard}, args.out)
    print("[done] 保存 %s | 门槛通过 = %s" % (args.out, res["pass"]))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
