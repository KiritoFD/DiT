#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""train_style_discriminator.py —— 风格判别器（**吃图片** ✓；标准字体做量大管饱的负样本 ✓）。

★ 一次被我改回来的弯路（记下 ✓）
    第一版我把图片 encode 成 VAE latent (4×32×32) 再送判别器 ✗ —— 用户指出"判别器应该对图片" ✓
    三个理由都支持改回像素 ✓:
      ① 判别器是**独立模型**，不需要与扩散的 latent 空间对齐 ✓
      ② 4×32×32 的 latent **已经丢掉细笔触** ✗（这正是我先前诊断出的瓶颈 ✓）
      ③ 白等 70 分钟 VAE 编码 7.3 万张 ✗，而字体图**本就是 256×256 PNG** ✓ 直接可用 ✓

★ 核心思想（用户）: **用标准字体作为量大管饱的负样本** ✓
    anchor  = 真迹图（书家 w、字 c）
    正样本  = 同书家、**异字**的真迹 ✓
    难负    = **同一个字 c** 的 N 个**标准字体**渲染图 ✓✓（同字 ⇒ 严格内容匹配 ✓）
              + 同字、其他书家的真迹 ✓
  ⇒ 之前只用书家时"同字异风格"候选**平均仅 2.4 个** ✗；加入 16 个字体后可到 **~16–25** ✓✓
  ⇒ 而且字体负样本**无限可再生**（不依赖真迹标注 ✓）⇒ 名副其实"量大管饱" ✓

★ 损失 = InfoNCE，**必须写全**: `logsumexp(pos, negs) − pos/τ` ✓
    （漏掉 `− pos/τ` 就退化成"把一切都推开" ✗ —— 仓库里已记过这个坑 ✓）

用法
    python tools/train_style_discriminator.py --steps 6000 --batch 96 \
        --font-csv data/font_slots/font_slots.csv --out assets/style_disc_v1.pt
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
from src.eval.style.gates import run_gates                                # noqa: E402

IMG = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def real_rows(csv_path, roots, max_rows=0):
    """真迹（按 (书家,书体) 槽位均衡 ✓ —— 直接截断只会拿到前 1–2 位书家 ✗）。"""
    by = defaultdict(list)
    for r in csv.DictReader(open(csv_path, encoding="utf-8-sig")):
        if (r.get("aug") or "").strip():
            continue
        p = resolve(r.get("image_path", ""), roots)
        if not p:
            continue
        by[(r.get("calligrapher") or "", r.get("script") or "")].append({
            "path": p, "writer": (r.get("calligrapher") or "").strip(),
            "script": (r.get("script") or "").strip(),
            "char": (r.get("character") or "").strip(),
            "work": work_id(r.get("src_image_path", ""), r.get("image_path", "")),
            "is_font": 0})
    out = [d for k in sorted(by) for d in by[k]]
    return out[:max_rows] if max_rows else out


def font_rows(font_csv, *, per_char=None, seed=0, chars_keep=None):
    """标准字体渲染图 = **负样本池** ✓（量大管饱 ✓）。"""
    if not os.path.isfile(font_csv):
        return []
    rows = list(csv.DictReader(open(font_csv, encoding="utf-8")))
    if chars_keep:
        rows = [r for r in rows if r.get("char") in chars_keep]
    rng = random.Random(seed)
    by = defaultdict(list)
    for r in rows:
        by[r.get("char")].append(r)
    out = []
    for c, rs in sorted(by.items()):
        rng.shuffle(rs)
        for r in (rs if not per_char else rs[:per_char]):
            out.append({"path": r["path"], "writer": "@font:" + r["font"], "script": "font",
                        "char": c, "work": "font:" + r["font"], "is_font": 1})
    return out


class ImgSet:
    """图片集（一次性读进内存 ✓；224×224×3 uint8 ⇒ 每万张约 0.15 GB ✓）。"""

    def __init__(self, rows, size=IMG, tag="", verbose=True):
        from PIL import Image
        self.rows = rows
        n = len(rows)
        self.arr = np.zeros((n, size, size, 3), np.uint8)
        t0 = time.time()
        for i, r in enumerate(rows):
            a = np.asarray(Image.open(r["path"]).convert("RGB"))
            if a.shape[0] != size or a.shape[1] != size:
                a = np.asarray(Image.fromarray(a).resize((size, size), Image.BICUBIC))
            self.arr[i] = a
            if verbose and i and i % 20000 == 0:
                print("  [img %s] %d/%d (%.0fs)" % (tag, i, n, time.time() - t0), flush=True)
        if verbose:
            print("  [img %s] %d 张 / %.2f GB (%.0fs)"
                  % (tag, n, self.arr.nbytes / 1e9, time.time() - t0), flush=True)

    def batch(self, idx, device):
        a = self.arr[idx]
        x = torch.from_numpy(a).to(device).permute(0, 3, 1, 2).float() / 255.0
        m = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
        s = torch.tensor(STD, device=device).view(1, 3, 1, 1)
        return (x - m) / s


class Enc(nn.Module):
    """图片 -> d 维 L2 归一化嵌入（resnet18 主干 ✓，从 ImageNet 预训练起 ✓）。"""

    def __init__(self, d=256, pretrained=True):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        base = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.body = nn.Sequential(*list(base.children())[:-1])       # -> (B,512,1,1)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(512, d))

    def emb(self, x):
        return F.normalize(self.head(self.body(x)), dim=-1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="exp-std/csv/train.csv")
    ap.add_argument("--font-csv", default="data/font_slots/font_slots.csv")
    ap.add_argument("--root", action="append", default=[])
    ap.add_argument("--out", default="assets/style_disc_v1.pt")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=96, help="总 batch（含字体部分 ✓）")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--n-font-neg", type=int, default=6,
                    help="每个 anchor 抽几个**同字标准字体**负样本 ✓（核心 ✓）")
    ap.add_argument("--font-frac", type=float, default=0.5, help="batch 里字体 anchor 占比 ✓")
    ap.add_argument("--font-per-char", type=int, default=4,
                    help="每字保留几张字体图 ✓（0=全部 7.3 万 ⇒ 内存/时间都大 ✗）")
    ap.add_argument("--max-real", type=int, default=0, help="0=全部 ✓")
    ap.add_argument("--img", type=int, default=IMG)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--n-pairs", type=int, default=1200)
    ap.add_argument("--bootstrap", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="")
    args = ap.parse_args()
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    roots = args.root or [".", "data", "exp-std"]

    R = real_rows(args.csv, roots, args.max_real)
    print("[data] 真迹 %d | 书家 %d | 字 %d"
          % (len(R), len({r['writer'] for r in R}), len({r['char'] for r in R})), flush=True)
    F_ = font_rows(args.font_csv, per_char=(args.font_per_char or None), seed=args.seed,
                   chars_keep={r["char"] for r in R})
    print("[data] 字体 %d | 字体数 %d | 字 %d"
          % (len(F_), len({r['writer'] for r in F_}), len({r['char'] for r in F_})), flush=True)
    if not F_:
        print("  ✗ 无字体样本 ⇒ 先跑 tools/build_font_slot_dataset.py ✓")
        return 3
    n_font = len({r['writer'] for r in F_})

    IR = ImgSet(R, size=args.img, tag="real")
    IF = ImgSet(F_, size=args.img, tag="font")

    by_char_F, by_writer = defaultdict(list), defaultdict(list)
    for i, r in enumerate(F_):
        by_char_F[r["char"]].append(i)
    for i, r in enumerate(R):
        by_writer[(r["writer"], r["script"])].append(i)
    hit = sum(1 for r in R if by_char_F.get(r["char"]))
    print("[pair] 真迹中**有同字字体负样本**的比例 = %.0f%% ✓（这就是'量大管饱' ✓）"
          % (100.0 * hit / max(1, len(R))), flush=True)
    print("[pair] 每 anchor 的字体负上限 = %d ✓（此前只用书家时平均 2.4 ✗）" % n_font, flush=True)

    model = Enc(d=args.d, pretrained=not args.no_pretrained).to(dev)
    print("[model] 参数 %s" % format(sum(p.numel() for p in model.parameters()), ","), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    rng = random.Random(args.seed)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        # ★ 结构对齐用户的话: "**标准字体作为负样本**" ✓
        #   ⇒ anchor **全是真迹** ✓；字体**只作为负样本**出现 ✓
        #   （我第一版让字体样本自己当 anchor ✗ ⇒ 它们没有正样本 ⇒ 张量形状对不上 ✓，
        #     而且那也偏离了"字体是负样本"的原意 ✓）
        ia = [rng.randrange(len(R)) for _ in range(args.batch)]
        xb = IR.batch(ia, dev)
        zb = model.emb(xb)

        # ① 正样本: 同书家**异字**的真迹 ✓
        pos_idx, has_pos = [], []
        for i in ia:
            r = R[i]
            cand = [j for j in by_writer[(r["writer"], r["script"])] if R[j]["char"] != r["char"]]
            if cand:
                pos_idx.append(rng.choice(cand))
                has_pos.append(True)
            else:
                pos_idx.append(i)
                has_pos.append(False)
        zp = model.emb(IR.batch(pos_idx, dev))

        # ② ★★ 核心: **同字标准字体**难负 ✓（量大管饱 ✓ —— 每个 anchor 最多 n_font 个 ✓）
        #   ⚠ R 与 F_ 的索引**必须分开取** ✗ —— 混用会拿到**别的字**的负样本 ✗✗，
        #     那正好毁掉"同字异风格"这个唯一有意义的约束 ✓
        neg_by_anchor = []
        for i in ia:
            cand = list(by_char_F.get(R[i]["char"], []))
            rng.shuffle(cand)
            neg_by_anchor.append(cand[:args.n_font_neg])
        n_neg = sum(len(v) for v in neg_by_anchor)
        sim_n = None
        if n_neg:
            n_max = max(len(v) for v in neg_by_anchor)
            flat = [j for v in neg_by_anchor for j in v]
            flat += [neg_by_anchor[0][0]] * (n_max * len(neg_by_anchor) - n_neg)
            zn = model.emb(IF.batch(flat, dev))
            sim_all = torch.einsum("bd,nkd->nk", zb, zn.view(len(neg_by_anchor), n_max, -1))
            mask = torch.zeros((len(neg_by_anchor), n_max), dtype=torch.bool, device=dev)
            for k, v in enumerate(neg_by_anchor):
                mask[k, :len(v)] = True
            sim_n = (sim_all / args.tau).masked_fill(~mask, float("-inf"))

        # ③ InfoNCE（写全 ✓）: 每行 = 1 正 + 若干同字字体负 ✓
        pos = (zb * zp).sum(-1, keepdim=True) / args.tau
        logits = torch.cat([pos, sim_n], 1) if sim_n is not None else pos
        loss = (torch.logsumexp(logits, dim=1) - pos.squeeze(1)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % max(1, args.steps // 8) == 0 or step == 1:
            print("  [step %5d] InfoNCE %.4f | 正样本 %d/%d | 负样本 %d | %.0fs"
                  % (step, float(loss), sum(has_pos), len(has_pos), n_neg, time.time() - t0),
                  flush=True)

    # ── 性能指标（同一套门槛 ✓ + 字体维度 ✓）
    @torch.no_grad()
    def emb_all(IS, bs=32):
        o = []
        for s in range(0, len(IS.arr), bs):
            o.append(model.emb(IS.batch(list(range(s, min(s + bs, len(IS.arr)))), dev)).cpu().numpy())
        return np.concatenate(o)

    print("\n===== 判别器性能指标 =====", flush=True)
    E_R = emb_all(IR)
    resR = run_gates(E_R, R, n_pairs=args.n_pairs, seed=args.seed, bootstrap=args.bootstrap)
    resF = run_gates(emb_all(IF), F_, n_pairs=min(args.n_pairs, 800), seed=args.seed,
                     bootstrap=min(args.bootstrap, 800))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"model": model.state_dict(), "d": args.d, "img": args.img,
                "writers": sorted({r["writer"] for r in R}),
                "fonts": sorted({r["writer"] for r in F_}),
                "gates_real": resR, "gates_font": resF,
                "n_font_neg": args.n_font_neg, "tau": args.tau}, args.out)
    print("[done] 保存 %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
