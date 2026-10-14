#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""train_style_discriminator.py —— 风格判别器 ✓

★ 核心思想（用户 ✓）: **用 std 标准字形当作独立"虚拟书家"槽位** ✓✓
    槽位 = { 王羲之·行, 王羲之·楷, ..., std·楷, std·行, std·隶 }
    anchor  = 真迹(书家 w, 书体 s, 字 c)
    正样本  = 同槽位(w×s)、异字 ✓
    难负    = 同字 c、**异槽位** ✓ —— 其他书家 ✓ + **std 字形** ✓✓
  ⇒ 模型学"这是书家写的 ≠ 这是标准字形" ✓✓
  ⇒ std 字形已在管线里 ✓（train.csv 的 std_path ✓ 100% 覆盖 ✓）不用下载字体 ✓✓
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

IMG = 160
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def _posix(p):
    return str(p).replace("\\", "/")


def real_rows(csv_path, roots, max_rows=0):
    by = defaultdict(list)
    for r in csv.DictReader(open(csv_path, encoding="utf-8-sig")):
        if (r.get("aug") or "").strip():
            continue
        p = resolve(_posix(r.get("image_path", "")), roots)
        if not p:
            continue
        by[(r.get("calligrapher") or "", r.get("script") or "")].append({
            "path": p, "writer": (r.get("calligrapher") or "").strip(),
            "script": (r.get("script") or "").strip(),
            "char": (r.get("character") or "").strip(),
            "work": work_id(r.get("src_image_path", ""), r.get("image_path", "")),
            "std": _posix(r.get("std_path", ""))})
    out = [d for k in sorted(by) for d in by[k]]
    return out[:max_rows] if max_rows else out


def std_rows(real):
    """每个唯一 (script, char) ⇒ 一行 std 样本 ✓（writer = 'std·{script}' ✓）。"""
    seen, out = set(), []
    for r in real:
        key = (r["script"], r["char"])
        if key in seen or not r.get("std") or not os.path.isfile(r["std"]):
            continue
        seen.add(key)
        out.append({"path": r["std"], "writer": f"std·{r['script']}",
                    "script": r["script"], "char": r["char"],
                    "work": "std", "is_std": 1})
    return out


class ImgSet:
    def __init__(self, rows, size=IMG, tag=""):
        from PIL import Image
        self.rows = rows
        self.arr = np.zeros((len(rows), size, size, 3), np.uint8)
        t0 = time.time()
        for i, r in enumerate(rows):
            a = np.asarray(Image.open(_posix(r["path"])).convert("RGB"))
            if a.shape[:2] != (size, size):
                a = np.asarray(Image.fromarray(a).resize((size, size), Image.BICUBIC))
            self.arr[i] = a
        print("  [img %s] %d 张 / %.2f GB (%.0fs)"
              % (tag, len(rows), self.arr.nbytes / 1e9, time.time() - t0), flush=True)

    def batch(self, idx, device):
        a = self.arr[idx]
        x = torch.from_numpy(a).to(device).permute(0, 3, 1, 2).float() / 255.0
        m = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
        s = torch.tensor(STD, device=device).view(1, 3, 1, 1)
        return (x - m) / s


class Enc(nn.Module):
    def __init__(self, d=256, pretrained=True):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        base = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.body = nn.Sequential(*list(base.children())[:-1])
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(512, d))

    def emb(self, x):
        return F.normalize(self.head(self.body(x)), dim=-1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="exp-std/csv/train.csv")
    ap.add_argument("--root", action="append", default=[])
    ap.add_argument("--out", default="assets/style_disc_std.pt")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=192)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--n-std-neg", type=int, default=1)
    ap.add_argument("--n-writer-neg", type=int, default=3)
    ap.add_argument("--max-real", type=int, default=0)
    ap.add_argument("--img", type=int, default=IMG)
    ap.add_argument("--n-pairs", type=int, default=1500)
    ap.add_argument("--bootstrap", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="")
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    roots = args.root or [".", "data", "exp-std"]

    R = real_rows(args.csv, roots, args.max_real)
    print("[data] 真迹 %d | 书家 %d" % (len(R), len({r['writer'] for r in R})), flush=True)
    S = std_rows(R)
    print("[data] std 字形 %d | 槽位 %d"
          % (len(S), len({r['writer'] for r in S})), flush=True)

    IR = ImgSet(R, size=args.img, tag="real")
    IS = ImgSet(S, size=args.img, tag="std")

    by_writer = defaultdict(list)
    by_char_R = defaultdict(list)     # char -> [real idx]（异书家负 ✓）
    by_char_S = defaultdict(list)     # char -> [std idx]（std 负 ✓）
    for i, r in enumerate(R):
        by_writer[(r["writer"], r["script"])].append(i)
        by_char_R[r["char"]].append(i)
    for i, r in enumerate(S):
        by_char_S[r["char"]].append(i)
    hit_s = sum(1 for r in R if by_char_S.get(r["char"]))
    print("[pair] 有 std 负 = %.0f%% ✓ | 每 anchor 负样本 = %d std + ≤%d writer ✓"
          % (100.0 * hit_s / len(R), args.n_std_neg, args.n_writer_neg), flush=True)

    model = Enc(d=args.d, pretrained=not args.no_pretrained).to(dev)
    print("[model] %s 参数" % format(sum(p.numel() for p in model.parameters()), ","), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    rng = random.Random(args.seed)
    t0 = time.time()

    for step in range(1, args.steps + 1):
        ia = [rng.randrange(len(R)) for _ in range(args.batch)]
        zb = model.emb(IR.batch(ia, dev))

        # ① 正样本: 同书家×同书体、异字 ✓
        pos_idx, pos_char, has_pos = [], [], []
        for i in ia:
            r = R[i]
            cand = [j for j in by_writer[(r["writer"], r["script"])]
                    if R[j]["char"] != r["char"]]
            if cand:
                p = rng.choice(cand)
                pos_idx.append(p)
                pos_char.append(R[p]["char"])
                has_pos.append(True)
            else:
                pos_idx.append(i)
                pos_char.append(r["char"])
                has_pos.append(False)
        zp = model.emb(IR.batch(pos_idx, dev))

        # ② ★★ 难负: 同字 c₂（正样本的字 ✓）、异槽位 ✓
        #    收集 (is_std, data_idx) 对 ✓ → **统一 padding 到 n_max** ✓
        neg_padded = []                # 每个 anchor: list of (is_std, idx) ✓
        for k, c2 in enumerate(pos_char):
            negs = []
            # std 负 ✓（核心 ✓）
            std_c = list(by_char_S.get(c2, []))
            rng.shuffle(std_c)
            for j in std_c[:args.n_std_neg]:
                negs.append((True, j))
            # 异书家真迹负 ✓
            wr = [j for j in by_char_R.get(c2, []) if R[j]["writer"] != R[ia[k]]["writer"]]
            rng.shuffle(wr)
            for j in wr[:args.n_writer_neg]:
                negs.append((False, j))
            if not negs:
                negs = [(False, ia[k])]      # 退化: 自身（不影响 ✓ 因为 pos 也在 logits ✓）
            neg_padded.append(negs)

        # ★ 统一 padding 到 n_max ✓（不足则重复第一个 ✓ ⇒ 每 anchor 负样本数恒等 ✓）
        n_max = max(len(v) for v in neg_padded)
        for v in neg_padded:
            while len(v) < n_max:
                v.append(v[0])

        # ③ 编码全部唯一负样本 ✓（分 R/S 两池 ✓）
        r_negs = sorted(set(j for negs in neg_padded for is_s, j in negs if not is_s))
        s_negs = sorted(set(j for negs in neg_padded for is_s, j in negs if is_s))
        E_rneg = model.emb(IR.batch(r_negs, dev)) if r_negs else None
        E_sneg = model.emb(IS.batch(s_negs, dev)) if s_negs else None
        r_map = {j: q for q, j in enumerate(r_negs)} if r_negs else {}
        s_map = {j: q for q, j in enumerate(s_negs)} if s_negs else {}

        # ④ 组装 (B, n_max, d) 负样本嵌入张量 ✓
        neg_emb = torch.zeros(args.batch, n_max, args.d, device=dev)
        for k, negs in enumerate(neg_padded):
            for m, (is_s, j) in enumerate(negs):
                if is_s and s_map:
                    neg_emb[k, m] = E_sneg[s_map[j]]
                elif not is_s and r_map:
                    neg_emb[k, m] = E_rneg[r_map[j]]

        # ⑤ InfoNCE ✓
        n_neg = args.batch * n_max
        pos_sim = (zb * zp).sum(-1, keepdim=True) / args.tau
        sim_n = (zb.unsqueeze(1) * neg_emb).sum(-1) / args.tau
        logits = torch.cat([pos_sim, sim_n], 1)
        loss = (torch.logsumexp(logits, dim=1) - pos_sim.squeeze(1)).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % max(1, args.steps // 8) == 0 or step == 1:
            print("  [step %5d] InfoNCE %.4f | 正 %d/%d | 负 %d | %.0fs"
                  % (step, float(loss.detach()), sum(has_pos), len(has_pos),
                     n_neg, time.time() - t0), flush=True)

    # ── 性能指标 ✓
    @torch.no_grad()
    def emb_all(IS_, bs=64):
        o = []
        for s in range(0, len(IS_.arr), bs):
            o.append(model.emb(IS_.batch(list(range(s, min(s + bs, len(IS_.arr)))), dev)).cpu().numpy())
        return np.concatenate(o)

    print("\n===== 判别器性能指标 =====", flush=True)
    E_R = emb_all(IR)
    std_rows_eval = [r for r in R if r.get("std")][:1500]
    IS_eval = ImgSet(std_rows_eval, size=args.img, tag="std_eval")
    E_std = emb_all(IS_eval)
    S_eval = [{"writer": f"std·{r['script']}", "script": r["script"],
               "char": r["char"], "work": "std"} for r in std_rows_eval]
    resR = run_gates(E_R, R, n_pairs=args.n_pairs, seed=args.seed, bootstrap=args.bootstrap)
    # std 维度: std 样本之间的 2AFC ✓（std·楷 之间应可区分 ✓）
    resS = run_gates(E_std, S_eval, n_pairs=min(args.n_pairs, 500), seed=args.seed,
                     bootstrap=min(args.bootstrap, 500))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"model": model.state_dict(), "d": args.d, "img": args.img,
                "gates_real": resR, "gates_std": resS,
                "n_std_neg": args.n_std_neg, "n_writer_neg": args.n_writer_neg,
                "tau": args.tau}, args.out)
    print("[done] 保存 %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
