#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""eval_gen_with_disc.py —— 用 CE 判别器评**生成图** ✓（判别器的真正用武之地 ✓）。

★ 要回答的问题
    v85A 的风格支路到底学到了什么？
      → 生成图的嵌入离**目标书家**近，还是离 **std** 近？ ✓✓
      → 判别器能把生成图**归对书家**吗（vs 1/23 随机 ✓）？ ✓

★ 评测内容（对每个生成集 ✓）
    ① K 路辨认: 生成图 → 23 个书家槽原型 ⇒ top1/top5（随机 = 1/23 ✓ 用户口径 ✓）
    ② 2AFC: 生成图(anchor) vs 同书家异字真迹(正) vs 异书家同字真迹(负)（随机 = 0.5 ✓）
    ③ **std-margin**: cos(gen, 目标书家原型) − cos(gen, std·同书体原型) ✓✓
       （>0 = 比标准字形更像目标书家 ✓ —— v85 的核心问题 ✓）
    ④ GT 对照: 用真迹同字图做同样的 ①②③ ✓（判别器对真图的 ceiling ✓）

★ 基线口径（用户指出 ✓）: 同时报 **1/23**（K 路 ✓）与 **0.5**（2AFC ✓）两种随机线 ✓
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.style.bank import resolve                                    # noqa: E402

IMG = 160
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def _posix(p):
    return str(p).replace("\\", "/")


class Enc(nn.Module):
    def __init__(self, d=256, n_cls=26):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        base = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.body = nn.Sequential(*list(base.children())[:-1])
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(512, d))
        self.cls = nn.Linear(d, n_cls)

    def emb(self, x):
        return F.normalize(self.head(self.body(x)), dim=-1)

    def forward(self, x):
        return self.cls(self.head(self.body(x)))


class ImgSet:
    def __init__(self, paths, size=IMG, tag=""):
        from PIL import Image
        self.paths = paths
        self.arr = np.zeros((len(paths), size, size, 3), np.uint8)
        for i, p in enumerate(paths):
            a = np.asarray(Image.open(_posix(p)).convert("RGB"))
            if a.shape[:2] != (size, size):
                a = np.asarray(Image.fromarray(a).resize((size, size), Image.BICUBIC))
            self.arr[i] = a
        print("  [img %s] %d 张" % (tag, len(paths)), flush=True)

    def emb_all(self, model, dev, bs=64):
        o = []
        with torch.no_grad():
            for s in range(0, len(self.arr), bs):
                x = self.batch(list(range(s, min(s + bs, len(self.arr)))), dev)
                o.append(model.emb(x).cpu().numpy())
        return np.concatenate(o)

    def batch(self, idx, device):
        a = self.arr[idx]
        x = torch.from_numpy(a).to(device).permute(0, 3, 1, 2).float() / 255.0
        m = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
        s = torch.tensor(STD, device=device).view(1, 3, 1, 1)
        return (x - m) / s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="assets/style_disc_ce_holdout.pt")
    ap.add_argument("--csv", default="exp-std/csv/eval200_fixed.csv")
    ap.add_argument("--root", action="append", default=[])
    ap.add_argument("--gen", action="append", default=[],
                    help="name=dir（含 g*.png ✓）可多次 ✓")
    ap.add_argument("--train-csv", default="exp-std/csv/train.csv",
                    help="建**书家原型库**用 ✓（判别器见过的图 ✓）")
    ap.add_argument("--out", default="assets/style_eval_local/disc_gen_eval.json")
    ap.add_argument("--device", default="")
    args = ap.parse_args()
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    roots = args.root or [".", "data", "exp-std"]

    # ── ① 判别器 ──
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    all_keys = ck["all_keys"]
    n_cls = ck["n_cls"]
    model = Enc(d=ck["d"], n_cls=n_cls).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()                                   # ★ BN running stats ✓
    writer_keys = [k for k in all_keys if not k[0].startswith("std·")]
    std_keys = [k for k in all_keys if k[0].startswith("std·")]
    print("[disc] %d 类 = %d 书家 + %d std" % (n_cls, len(writer_keys), len(std_keys)),
          flush=True)
    key2cls = {k: i for i, k in enumerate(all_keys)}

    # ── ② 书家原型库（从训练 CSV ✓ 判别器见过的图 ✓）──
    import random
    rng = random.Random(0)
    by_slot = defaultdict(list)
    std_by_slot = defaultdict(list)
    tr = list(csv.DictReader(open(args.train_csv, encoding="utf-8-sig")))
    rng.shuffle(tr)
    tr = tr[:6000]                                 # 每槽限流 ✓（够稳 ✓）
    tr_paths, tr_meta = [], []
    for r in tr:
        if (r.get("aug") or "").strip():
            continue
        p = resolve(_posix(r.get("image_path", "")), roots)
        if not p:
            continue
        w, s, c = (r.get("calligrapher") or "").strip(), (r.get("script") or "").strip(), \
            (r.get("character") or "").strip()
        tr_paths.append(p)
        tr_meta.append((w, s, c))
        by_slot[(w, s)].append(len(tr_paths) - 1)
        sp = resolve(_posix(r.get("std_path", "")), roots)
        if sp and s and c and (s, c) not in {x for x in
                                             [(m[1], m[2]) for m in std_by_slot.get("std·" + s, [])]}:
            pass
    IS_tr = ImgSet(tr_paths, tag="bank")
    E_tr = IS_tr.emb_all(model, dev)
    protos = {}
    for k, idxs in sorted(by_slot.items()):
        protos[k[0] + u'·' + k[1]] = E_tr[idxs].mean(0)   # 字符串键 ✓
    protos = {k: v / np.linalg.norm(v) for k, v in protos.items()}
    print("[bank] %d 槽原型 ✓（来自 %d 张训练图 ✓）" % (len(protos), len(tr_paths)), flush=True)

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8-sig")))

    # ── ②-b std 原型（用 **评测 CSV 的 std_path** ✓ —— 正是所评字符的标准字形 ✓）
    std_paths, std_seen = [], set()
    for r in rows:
        s, c = (r.get("script") or "").strip(), (r.get("character") or "").strip()
        key = (s, c)
        if key in std_seen:
            continue
        sp = resolve(_posix(r.get("std_path", "")), roots)
        if sp and s:
            std_seen.add(key)
            std_paths.append((sp, s))
    std_by_slot = defaultdict(list)
    if std_paths:
        IS_std = ImgSet([p for p, _ in std_paths], tag="std")
        E_std = IS_std.emb_all(model, dev)
        for q, (_p, s) in enumerate(std_paths):
            std_by_slot["std·" + s].append(q)
        for k, idxs in std_by_slot.items():
            v = E_std[idxs].mean(0)
            protos[k] = v / np.linalg.norm(v)
    print("[bank] std 原型 %d 槽 ✓（%d 张 ✓）" % (len(std_by_slot), len(std_paths)), flush=True)

    # ── ③ 生成图 + GT ──
    print("[eval-csv] %d 行" % len(rows), flush=True)

    def load_gen(d):
        fs = [f for f in os.listdir(d) if re.match(r"^g\d+\.png$", f)]
        fs.sort(key=lambda f: int(re.findall(r"\d+", f)[0]))
        return [os.path.join(d, f) for f in fs]

    # GT 真迹（同 CSV ✓ —— 判别器 ceiling ✓）
    gt_paths = [resolve(_posix(r.get("image_path", "")), roots) for r in rows]
    gt_ok = [(i, p) for i, p in enumerate(gt_paths) if p]
    gt_meta = [(r.get("calligrapher", "").strip(), r.get("script", "").strip(),
                r.get("character", "").strip()) for r in rows]

    results = {}
    # GT 先评 ✓（ceiling ✓）
    sets = [("GT真迹", None)] + [(n, d) for n, d in
                                 (s.split("=", 1) for s in args.gen if "=" in s)]
    for name, d in sets:
        if d is None:
            pairs = gt_ok
            IS = ImgSet([p for _, p in pairs], tag=name)
        else:
            ps = load_gen(d)
            if len(ps) != len(rows):
                print("  ⚠ %s: %d 张 ≠ CSV %d 行 ⇒ 按序截齐 ✓" % (name, len(ps), len(rows)))
            pairs = list(enumerate(ps[:len(rows)]))
            IS = ImgSet([p for _, p in pairs], tag=name)
        E = IS.emb_all(model, dev)
        E = E / np.linalg.norm(E, axis=1, keepdims=True)
        P = np.stack([protos[k] for k in sorted(protos)])          # (n_slot, d)
        slot_list = sorted(protos)
        sims = E @ P.T                                             # (n, n_slot)

        top1 = top5 = n_ok = 0
        std_margins, writer_margins = [], []
        slot_script = {k: k.split(chr(183))[1] if chr(183) in k else '' for k in slot_list}
        for j, (row_i, _p) in enumerate(pairs):
            w, s, c = gt_meta[row_i]
            key = w + chr(183) + s
            if key not in protos:
                continue
            n_ok += 1
            ci = slot_list.index(key)
            order = np.argsort(-sims[j])
            rank = int(np.where(order == ci)[0][0])
            top1 += int(rank == 0)
            top5 += int(rank < 5)
            # std 同书体原型 ✓
            sk = "std·" + s
            if sk in protos:
                std_margins.append(float(sims[j][ci] - sims[j][slot_list.index(sk)]))
            others = [sims[j][k2] for k2 in range(len(slot_list))
                      if k2 != ci and slot_script[slot_list[k2]] == s]  # 同书体异书家 ✓
            if others:
                writer_margins.append(float(sims[j][ci] - max(others)))
        res = {"n": n_ok, "top1": top1 / max(1, n_ok), "top5": top5 / max(1, n_ok),
               "chance_top1": 1.0 / len(writer_keys),
               "std_margin_mean": float(np.mean(std_margins)) if std_margins else None,
               "writer_margin_mean": float(np.mean(writer_margins)) if writer_margins else None}
        results[name] = res
        print("  [%s] top1 %.4f (随机 1/%d=%.4f) | top5 %.4f | std-margin %s | writer-margin %s"
              % (name, res["top1"], len(writer_keys), res["chance_top1"], res["top5"],
                 ("%.4f" % res["std_margin_mean"]) if res["std_margin_mean"] is not None else "NA",
                 ("%.4f" % res["writer_margin_mean"]) if res["writer_margin_mean"] is not None
                 else "NA"), flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    import json
    json.dump(results, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("[done] %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
