#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""train_style_discriminator.py —— 风格判别器（**纯 CrossEntropy 分类** ✓）。

★ 为什么改用 CE（用户指出 ✓）:
    "这是一个判别或者说分类任务啊，为什么会只挂着一个 infoNCE 呢" ✓✓
    ⇒ 任务本质 = 10 类分类（谁写的）+ 3 个 std 虚拟书家类 ✓
    ⇒ InfoNCE 是无监督表示学习 ✗ 绕远路 ✗
    ⇒ **CE 直接优化"分对书家"** ✓✓
    ⇒ 仓库反例: cls_style_v1 用 CE 拿到 Macro-F1 **0.746** ✓✓ —— CE 明明有效 ✓

★ 设计
    模型: resnet18(ImageNet ✓) → 256D L2 嵌入 → Linear(256, n_cls) ✓
    类别: 10 真书家 + 3 std 槽（std·楷/std·行/std·隶 ✓）= **13 类** ✓
    损失: **CrossEntropy** ✓（纯 CE ✓ 无 margin ✓ 先看基线 ✓）
    评测: 同一套门槛 ✓（G1 2AFC / G2 mAP ✓ 用 **L2 嵌入**（分类头之前 ✓））
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
    """resnet18 → 256D L2 嵌入（评测用 ✓）→ Linear(n_cls) 分类头（训练用 ✓）。"""

    def __init__(self, d=256, n_cls=13, pretrained=True):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        base = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.body = nn.Sequential(*list(base.children())[:-1])
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(512, d))
        self.cls = nn.Linear(d, n_cls)

    def emb(self, x):
        """L2 归一化嵌入（**评测用** ✓ —— 分类头之前 ✓）。"""
        return F.normalize(self.head(self.body(x)), dim=-1)

    def forward(self, x):
        e = self.head(self.body(x))          # 未归一化 ✓（CE 用 ✓）
        return self.cls(e)                   # logits ✓


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="exp-std/csv/train.csv")
    ap.add_argument("--root", action="append", default=[])
    ap.add_argument("--out", default="assets/style_disc_ce.pt")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=192)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--max-real", type=int, default=0)
    ap.add_argument("--img", type=int, default=IMG)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--label-smooth", type=float, default=0.0)
    ap.add_argument("--n-pairs", type=int, default=1500)
    ap.add_argument("--bootstrap", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout-work-frac", type=float, default=0.3,
                    help="**作品级留出比例** ✓（训练只见其余 ✓ —— 上一版 0.9987 是记忆 ✗）")
    ap.add_argument("--device", default="")
    args = ap.parse_args()
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    roots = args.root or [".", "data", "exp-std"]

    # ── 数据 ──
    R_all = real_rows(args.csv, roots, args.max_real)
    S = std_rows(R_all)
    print("[data] 真迹 %d | std %d" % (len(R_all), len(S)), flush=True)

    # ★★ **作品级 train/eval 划分** ✓（真正的泛化检验 ✓）
    #    上一版在全部作品上训练 ⇒ gates 的 0.9987 是**记忆**不是泛化 ✗✗
    #    本版: 训练只见 (1-frac) 的作品 ✓ 评测在**从未见过的** frac 作品上 ✓
    import random as _rnd
    _wr = _rnd.Random(args.seed)
    works = sorted({r["work"] for r in R_all})
    _wr.shuffle(works)
    n_hold = max(1, int(len(works) * args.holdout_work_frac))
    hold_works = set(works[:n_hold])
    R = [r for r in R_all if r["work"] not in hold_works]
    R_eval = [r for r in R_all if r["work"] in hold_works]
    print("[split] 作品 %d 个 ⇒ 训练 %d / 留出 %d（真迹 %d/%d 张 ✓）"
          % (len(works), len(works) - n_hold, n_hold, len(R), len(R_eval)), flush=True)

    # 类别表: 真书家（按槽位 ✓）+ std 槽 ✓（用 **R_all** ✓ 防"某书家只在留出集"✗）
    writer_keys = sorted({(r["writer"], r["script"]) for r in R_all})
    std_keys = sorted({(r["writer"], r["script"]) for r in S})
    all_keys = writer_keys + std_keys
    key2cls = {k: i for i, k in enumerate(all_keys)}
    n_cls = len(all_keys)
    print("[cls] %d 类 = %d 真书家 + %d std 槽" % (n_cls, len(writer_keys), len(std_keys)),
          flush=True)
    for k, v in list(key2cls.items())[:13]:
        print("    %2d ← %s·%s" % (v, k[0], k[1]))

    IR = ImgSet(R, size=args.img, tag="real")
    IS = ImgSet(S, size=args.img, tag="std")
    cls_R = torch.tensor([key2cls[(r["writer"], r["script"])] for r in R],
                         dtype=torch.long, device=dev)
    cls_S = torch.tensor([key2cls[(r["writer"], r["script"])] for r in S],
                         dtype=torch.long, device=dev)

    # ── 训练 ──
    model = Enc(d=args.d, n_cls=n_cls, pretrained=not args.no_pretrained).to(dev)
    print("[model] %s 参数" % format(sum(p.numel() for p in model.parameters()), ","), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)
    rng = random.Random(args.seed)
    n_all = len(R) + len(S)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = [rng.randrange(n_all) for _ in range(args.batch)]
        # 混合采样 ✓（真迹 + std ✓ —— CE 天然支持 ✓）
        r_idx = [i for i in idx if i < len(R)]
        s_idx = [i - len(R) for i in idx if i >= len(R)]
        parts, labels = [], []
        if r_idx:
            parts.append(IR.batch(r_idx, dev))
            labels.append(cls_R[r_idx])
        if s_idx:
            parts.append(IS.batch(s_idx, dev))
            labels.append(cls_S[s_idx])
        x = torch.cat(parts, 0)
        y = torch.cat(labels, 0)
        logits = model(x)
        loss = crit(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % max(1, args.steps // 10) == 0 or step == 1:
            print("  [step %5d] CE %.4f | acc %.3f | %.0fs"
                  % (step, float(loss.detach()), float(acc), time.time() - t0), flush=True)

    # ── 评测（用 L2 嵌入 ✓ 同一套门槛 ✓）
    # ★★ 必须 model.eval() ✓（实测大坑 ✓）:
    #    train 模式下 BatchNorm 用**当前 batch 统计** ✓ 而评测 batch 按 (书家,书体)
    #    排序 ⇒ 每个 batch 全是同一位书家 ⇒ BN 统计被同质 batch 扭曲 ✗✗
    #    ⇒ 实测: 训练 acc 0.974 但评测 acc 0.06（≈随机 ✗）⇒ 嵌入全是垃圾 ✗
    #    eval 模式用**累积的 running stats** ✓ ⇒ 稳定 ✓
    model.eval()
    print("[eval] model.eval() ✓（BN 用 running stats ✓）", flush=True)
    @torch.no_grad()
    def emb_all(IS_, bs=64):
        o = []
        for s in range(0, len(IS_.arr), bs):
            o.append(model.emb(IS_.batch(list(range(s, min(s + bs, len(IS_.arr)))), dev))
                     .cpu().numpy())
        return np.concatenate(o)

    print("\n===== 判别器性能指标（CE 版 ✓ **作品级留出** ✓）=====", flush=True)
    # ★ 评测在**从未见过的作品**上 ✓（R_eval ✓ —— 记忆不算数 ✗）
    IE = ImgSet(R_eval, size=args.img, tag="eval")
    cls_RE = torch.tensor([key2cls[(r["writer"], r["script"])] for r in R_eval],
                          dtype=torch.long, device=dev)
    E_E = emb_all(IE)
    resR = run_gates(E_E, R_eval, n_pairs=args.n_pairs, seed=args.seed,
                     bootstrap=args.bootstrap)
    resS = run_gates(emb_all(IS), S, n_pairs=min(args.n_pairs, 500),
                     seed=args.seed, bootstrap=min(args.bootstrap, 500))
    # ★ 分类准确率: **留出作品上** ✓（真正的泛化指标 ✓）
    @torch.no_grad()
    def classify(IS_, labels, bs=64):
        correct = 0
        for s in range(0, len(IS_.arr), bs):
            idx = list(range(s, min(s + bs, len(IS_.arr))))
            pred = model(IS_.batch(idx, dev)).argmax(1)
            correct += int((pred == labels[idx]).sum())
        return correct / len(IS_.arr)

    acc_train = classify(IR, cls_R)
    acc_eval = classify(IE, cls_RE)
    acc_S = classify(IS, cls_S)
    print("  分类 acc: 训练作品 %.4f | **留出作品 %.4f** | std %.4f"
          % (acc_train, acc_eval, acc_S), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"model": model.state_dict(), "d": args.d, "img": args.img,
                "n_cls": n_cls, "all_keys": all_keys,
                "acc_train": acc_train, "acc_eval": acc_eval, "acc_std": acc_S,
                "holdout_works": sorted(hold_works),
                "gates_eval": resR, "gates_std": resS}, args.out)
    print("[done] 保存 %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
