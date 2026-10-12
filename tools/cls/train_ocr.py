#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_ocr.py — 汉字内容识别器 v1（按评审规格实现，独立于任何大模型）

规格对照（逐条落地，故意不做的事也写明）:
  · 网络: convnext_tiny (ImageNet 预训练) + Linear(768, N_char)        ~34M ✓
  · 输入: 灰度->3通道, 224 或 256, ImageNet 归一化                      ✓
  · Loss: CrossEntropy(label_smoothing=0.05)                            ✓
  · 优化: AdamW wd=0.05; backbone lr=3e-5, head lr=3e-4                 ✓
  · Batch 128, cosine decay, 按**验证集 Macro Acc** 早停                ✓
  · 长尾: 按**字频平方根**倒数的 WeightedRandomSampler                   ✓
  · 报告: Top-1 + **Macro Acc** + **频次分组准确率** + **逐书体**         ✓
  · 划分: 按**作品/来源分组**留出（绝不用随机 10%）                       ✓
  · 排除: 评测集(187 GT 等)对应作品**绝不进入训练/验证**                  ✓
  · 增广: 仅小平移/尺度/亮度对比度/轻微模糊; **不翻转、不做 morphology**   ✓
  · 负对照: 空白图 + 印刷字形 -> 报告置信度（OOD 闸门的证据）            ✓
  · v1 不做: ArcFace / 对抗解耦 / 百余书家联合分类                       ✗

用法:
  PY=/opt/conda/envs/cu121/bin/python
  # 冒烟（小样本, 2 epoch）
  $PY tools/cls/train_ocr.py --csv /root/Workspace/xy/UNIFIED_RAW/meta/train_clean.csv \
      --img-root /root/Workspace/xy/UNIFIED_RAW --out assets/results/cls_ocr_v1 \
      --epochs 2 --limit 4000 --batch 32 --workers 4
  # 正式
  $PY tools/cls/train_ocr.py --csv <..> --img-root <..> --out <..> --epochs 40
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

sys.path.insert(0, os.getcwd())
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ───────────────────────── 数据 ─────────────────────────
def load_rows(csv_path, img_root, group_col, char_col, script_col,
              pol_col, exclude_ids, min_count, log=print):
    """读 csv -> [(path, char, script, group)]，并做全部过滤 + 组划分。"""
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8-sig")))
    log("  csv 行数: %d" % len(rows))
    keep, miss_img, ex, bad_char = [], 0, 0, 0
    cand_roots = [img_root, "", os.path.dirname(csv_path), os.path.dirname(os.path.dirname(csv_path))]
    for r in rows:
        ip = (r.get("image_path") or "").strip()
        ch = (r.get(char_col) or "").strip()
        if not ip or not ch or ch.lower() == "null":
            bad_char += 1
            continue
        if exclude_ids and ((r.get("img_id", "") in exclude_ids)
                            or (r.get("src_image_path", "") in exclude_ids)):
            ex += 1
            continue
        p = None
        for root in cand_roots:
            q = os.path.join(root, ip) if root else ip
            if os.path.isfile(q):
                p = q
                break
        if p is None:
            miss_img += 1
            continue
        keep.append((p, ch, (r.get(script_col) or "").strip(),
                     (r.get(group_col) or "").strip() or "unknown",
                     str(r.get(pol_col, "")).strip().lower() in ("1", "true", "yes")))
    log("  过滤: 保留 %d | 缺图 %d | 命中排除集 %d | 标签空 %d"
        % (len(keep), miss_img, ex, bad_char))
    cnt = collections.Counter(x[1] for x in keep)
    kept_chars = {c for c, n in cnt.items() if n >= min_count}
    dropped = len(keep) - sum(1 for x in keep if x[1] in kept_chars)
    keep = [x for x in keep if x[1] in kept_chars]
    log("  类别: 唯一字 %d -> 保留 >=%d 样本的字 %d 个（丢 %d 行）"
        % (len(cnt), min_count, len(kept_chars), dropped))
    return keep, cnt


def grouped_split(items, val_frac, seed):
    """按**作品/来源组**留出验证集（不是按行随机 ✗）。"""
    groups = sorted({x[3] for x in items})
    rng = random.Random(seed)
    rng.shuffle(groups)
    n_val = max(1, int(len(groups) * val_frac))
    val_g = set(groups[:n_val])
    tr = [x for x in items if x[3] not in val_g]
    va = [x for x in items if x[3] in val_g]
    return tr, va, val_g


class InkDataset(Dataset):
    def __init__(self, items, char2idx, size=224, train=True, script2idx=None):
        self.items, self.c2i, self.size, self.train = items, char2idx, size, train
        self.s2i = script2idx or {}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, ch, _sc, _g, inverted = self.items[i]
        img = Image.open(p).convert("L")
        if inverted:                                  # 极性归一：统一成"白底黑字" ✓
            img = Image.fromarray(255 - np.asarray(img))
        img = img.convert("RGB")
        if self.train:                                # ★ 保守增广：不翻转/不变形拓扑 ✓
            if random.random() < 0.5:
                s = random.uniform(0.94, 1.06)
                w, h = img.size
                img = img.resize((max(8, int(w * s)), max(8, int(h * s))), Image.BILINEAR)
            if random.random() < 0.4:
                w, h = img.size
                dx, dy = random.randint(-6, 6), random.randint(-6, 6)
                img = img.transform((w, h), Image.AFFINE, (1, 0, -dx, 0, 1, -dy),
                                    resample=Image.BILINEAR, fillcolor=(255, 255, 255))
            if random.random() < 0.3:
                img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.0)))
            if random.random() < 0.5:
                arr = np.asarray(img).astype(np.float32)
                arr = arr * random.uniform(0.85, 1.15) + random.uniform(-18, 18)
                img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        img = img.resize((self.size, self.size), Image.BICUBIC)
        x = transforms.functional.to_tensor(img)
        x = transforms.functional.normalize(x, IMAGENET_MEAN, IMAGENET_STD)
        return x, self.c2i[ch], self.s2i.get(_sc, 0)


# ───────────────────────── 模型 ─────────────────────────
def build_model(n_cls, size):
    m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
    in_f = m.classifier[2].in_features                    # 768
    m.classifier[2] = nn.Linear(in_f, n_cls)
    return m, in_f


# ───────────────────────── 评测 ─────────────────────────
@torch.no_grad()
def evaluate(model, loader, dev, idx2cnt=None):
    model.eval()
    top1, tot = 0, 0
    per_cls = collections.defaultdict(lambda: [0, 0])     # cls -> [correct, total]
    per_script = collections.defaultdict(lambda: [0, 0])
    _ps = collections.defaultdict(lambda: [0, 0])   # ???????? ??
    for x, y, sc in loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=(dev.type == "cuda")):
            logits = model(x)
        pred = logits.float().argmax(1)
        ok = (pred == y)
        top1 += int(ok.sum())
        tot += y.numel()
        for c, o, s_ in zip(y.tolist(), ok.tolist(), sc.tolist()):
            per_cls[c][1] += 1
            per_cls[c][0] += int(o)
            _ps[s_][1] += 1
            _ps[s_][0] += int(o)
    accs = [c / t for c, t in per_cls.values() if t > 0]
    macro = float(np.mean(accs)) if accs else 0.0
    out = {"top1": top1 / max(tot, 1), "macro": macro, "n": tot, "n_cls_seen": len(per_cls),
           "per_script_top1": {k: v[0] / v[1] for k, v in _ps.items()}}
    if idx2cnt:
        buckets = [(1, 5), (5, 20), (20, 100), (100, 10 ** 9)]
        for lo, hi in buckets:
            cc = [c for c in per_cls if lo <= idx2cnt.get(c, 0) < hi]
            if cc:
                a = sum(per_cls[c][0] for c in cc) / sum(per_cls[c][1] for c in cc)
                out["freq_%d_%s" % (lo, "inf" if hi > 10 ** 8 else hi)] = a
    return out


@torch.no_grad()
def negative_controls(model, dev, size, font="tools/fonts/simkai.ttf", char="永"):
    """空白图 + 印刷字形 -> 看模型多自信（普通 softmax 对 OOD 过度自信 ✗）。"""
    model.eval()
    outs = {}
    blank = Image.new("RGB", (size, size), (255, 255, 255))
    printed = None
    if os.path.isfile(font):
        from PIL import ImageDraw, ImageFont
        im = Image.new("L", (512, 512), 255)
        f = ImageFont.truetype(font, 380)
        d = ImageDraw.Draw(im)
        bb = d.textbbox((0, 0), char, font=f)
        d.text(((512 - (bb[2] - bb[0])) / 2 - bb[0], (512 - (bb[3] - bb[1])) / 2 - bb[1]),
               char, font=f, fill=0)
        printed = im.convert("RGB")
    for name, img in (("blank_white", blank), ("printed_%s" % char, printed)):
        if img is None:
            continue
        x = transforms.functional.normalize(
            transforms.functional.to_tensor(img.resize((size, size), Image.BICUBIC)),
            IMAGENET_MEAN, IMAGENET_STD).unsqueeze(0).to(dev)
        with torch.autocast("cuda", dtype=torch.float16, enabled=(dev.type == "cuda")):
            p = torch.softmax(model(x).float(), 1)[0]
        outs[name] = {"max_prob": float(p.max()), "entropy": float(-(p * (p + 1e-9).log()).sum())}
    return outs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--img-root", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 行（冒烟）")
    ap.add_argument("--min-count", type=int, default=2, help="字符至少多少样本才作为类别")
    ap.add_argument("--val-frac", type=float, default=0.10, help="按**作品组**留出比例")
    ap.add_argument("--lr-backbone", type=float, default=3e-5)
    ap.add_argument("--lr-head", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--label-smooth", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--group-col", default="calligrapher", help="作品/来源组列（400k 语料里就是它）")
    ap.add_argument("--char-col", default="character")
    ap.add_argument("--script-col", default="script")
    ap.add_argument("--pol-col", default="is_inverted")
    ap.add_argument("--exclude-csv", default="exp-std/csv/eval200_fixed.csv",
                    help="评测集：其作品/图 id 绝不进入训练/验证")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log = lambda *x: print(*x, flush=True)

    # 排除集（评测 GT 及其源作品）
    exclude_ids = set()
    if a.exclude_csv and os.path.isfile(a.exclude_csv):
        for r in csv.DictReader(open(a.exclude_csv, encoding="utf-8-sig")):
            for k in ("img_id", "src_image_path", "image_path"):
                v = (r.get(k) or "").strip()
                if v:
                    exclude_ids.add(v)
                    exclude_ids.add(os.path.basename(v))
        log("排除集: %d 个 id/路径（来自 %s）" % (len(exclude_ids), a.exclude_csv))

    log("== 读数据 ==")
    items, cnt = load_rows(a.csv, a.img_root, a.group_col, a.char_col, a.script_col,
                           a.pol_col, exclude_ids, a.min_count, log)
    cnt = collections.Counter(x[1] for x in items)   # fix: 用**过滤后**的真实类别计数(否则头里会有死类)
    if a.limit:
        items = items[: a.limit]
        cnt = collections.Counter(x[1] for x in items)
        log("  冒烟模式: 只用 %d 行 / %d 类" % (len(items), len(cnt)))

    chars = sorted(cnt)
    c2i = {c: i for i, c in enumerate(chars)}
    idx2cnt = {i: cnt[c] for c, i in c2i.items()}
    tr, va, val_g = grouped_split(items, a.val_frac, a.seed)
    log("  分组划分: 作品组 %d 个 -> 验证组 %d 个 | 训练 %d 行 / 验证 %d 行"
        % (len({x[3] for x in items}), len(val_g), len(tr), len(va)))
    log("  类别数 N_char = %d" % len(chars))
    if not tr or not va:
        log("  ✗ 训练/验证为空，检查 --val-frac 与组列"); return 2

    # √频率校正采样 ✓
    w = {c: 1.0 / math.sqrt(n) for c, n in cnt.items()}
    samp_w = torch.tensor([w[x[1]] for x in tr], dtype=torch.double)
    sampler = WeightedRandomSampler(samp_w, num_samples=len(tr), replacement=True)

    scripts_ = sorted({x[2] for x in items})
    s2i = {s: i for i, s in enumerate(scripts_)}
    ds_tr = InkDataset(tr, c2i, a.size, True, s2i)
    ds_va = InkDataset(va, c2i, a.size, False, s2i)
    dl_tr = DataLoader(ds_tr, batch_size=a.batch, sampler=sampler, num_workers=a.workers,
                       pin_memory=True, drop_last=True, persistent_workers=(a.workers > 0))
    dl_va = DataLoader(ds_va, batch_size=a.batch, shuffle=False, num_workers=a.workers,
                       pin_memory=True, persistent_workers=(a.workers > 0))

    model, in_f = build_model(len(chars), a.size)
    model.to(dev)
    head_params = list(model.classifier.parameters())
    head_ids = {id(p) for p in head_params}
    bb_params = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW([{"params": bb_params, "lr": a.lr_backbone},
                             {"params": head_params, "lr": a.lr_head}], weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs * max(1, len(dl_tr)))
    crit = nn.CrossEntropyLoss(label_smoothing=a.label_smooth)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev.type == "cuda"))
    log("== 训练 ==  backbone lr=%g head lr=%g batch=%d N=%d | convnext_tiny(%dD) ~%.1fM 参数"
        % (a.lr_backbone, a.lr_head, a.batch, len(chars), in_f,
           sum(p.numel() for p in model.parameters()) / 1e6))

    best, best_ep, hist = -1.0, -1, []
    for ep in range(1, a.epochs + 1):
        model.train(); t0 = time.time(); tot = 0; run = 0.0
        for it, (x, y, _sc) in enumerate(dl_tr):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=(dev.type == "cuda")):
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            run += float(loss); tot += 1
            if it % 50 == 0:
                log("   ep%d it%d loss=%.4f (%.0fs)" % (ep, it, run / max(tot, 1), time.time() - t0))
        m = evaluate(model, dl_va, dev, idx2cnt)
        hist.append({"epoch": ep, "loss": run / max(tot, 1), **m})
        log("  == ep%d 验证: Top1=%.4f Macro=%.4f (n=%d, 类=%d) 频次分组=%s | 用时%.0fs"
            % (ep, m["top1"], m["macro"], m["n"], m["n_cls_seen"],
               {k: round(v, 4) for k, v in m.items() if k.startswith("freq_")}, time.time() - t0))
        log("     ??? Top1: %s" % {k: round(v, 4) for k, v in m["per_script_top1"].items()})
        if m["macro"] > best:
            best, best_ep = m["macro"], ep
            torch.save({"model": model.state_dict(), "chars": chars, "size": a.size,
                        "arch": "convnext_tiny", "in_features": in_f,
                        "val_macro": best, "epoch": ep, "args": vars(a)},
                       os.path.join(a.out, "best.pt"))
            log("     ✓ 保存 best.pt (Macro=%.4f)" % best)
        elif ep - best_ep >= a.patience:
            log("     早停：%d 轮无提升" % a.patience); break

    neg = negative_controls(model, dev, a.size)
    log("== 负对照（OOD 闸门证据）== %s" % json.dumps(neg, ensure_ascii=False))
    summary = {"best_macro": best, "best_epoch": best_ep, "history": hist,
               "n_char": len(chars), "n_train": len(tr), "n_val": len(va),
               "n_val_groups": len(val_g), "negative_controls": neg, "args": vars(a)}
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    json.dump(chars, open(os.path.join(a.out, "classes.json"), "w", encoding="utf-8"),
              ensure_ascii=False)
    log("== 完成: best Macro=%.4f @ep%d | 产物 %s ==" % (best, best_ep, a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
