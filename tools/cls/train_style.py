#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_style.py — 书家识别 + 书体识别 v1（共享轻量 backbone，双头）

规格对照:
  · backbone: resnet18 (ImageNet) -> fc=Identity -> 512D                     ✓
  · projector: Linear(512,256) + LayerNorm + ReLU                            ✓
  · heads: writer_head(256->10), script_head(256->3)                         ✓
  · Loss: L_writer + 0.2 * L_script   (书体只作辅助；若制造虚假关联则拆开)     ✓
  · 数据: Top10 的 **26,002 真实图**，`aug` 增广副本**不当独立样本**           ✓
  · 优化: AdamW wd=0.01; backbone 2e-5, head 2e-4; batch 64-128; 20-40 ep     ✓
  · 选模: **书家 Macro Acc**                                                  ✓
  · 关键诊断（防"看书体猜书家"）:
       - **同书体内**的书家准确率（逐书体）                                    ✓
       - **只看书体的基线** = 各书体内多数类作家准确率                          ✓
       - **同字×同书体×异书家** 对子的判别准确率                                ✓
  · 划分: 按**作品来源**(src_image_path 所在目录)留出，不随机抽 ✗                ✓
  · 排除: 评测集 187 GT 及其源作品不进训练/验证                                ✓
  · 负对照: 空白图 + 印刷字形 -> 置信度（OOD 闸门）                            ✓
  · v1 不做: ArcFace / 对抗解耦 / 百余书家联合分类                            ✗
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


def load_rows(csv_path, img_root, exclude_ids, drop_aug, log=print):
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8-sig")))
    log("  csv 行数: %d" % len(rows))
    aug_vals = collections.Counter(str(r.get("aug", "")).strip() for r in rows)
    log("  aug 取值分布: %s" % aug_vals.most_common(6))
    keep, miss, ex, aug_drop = [], 0, 0, 0
    roots = [img_root, "", os.path.dirname(csv_path), os.path.dirname(os.path.dirname(csv_path))]
    for r in rows:
        ip = (r.get("image_path") or "").strip()
        cw = (r.get("calligrapher") or "").strip()
        sc = (r.get("script") or "").strip()
        ch = (r.get("character") or "").strip()
        if not ip or not cw or not sc or not ch:
            continue
        if exclude_ids and ((r.get("img_id", "") in exclude_ids)
                            or (r.get("src_image_path", "") in exclude_ids)
                            or (os.path.basename(r.get("src_image_path") or "") in exclude_ids)):
            ex += 1
            continue
        if drop_aug and str(r.get("aug", "")).strip().lower() not in ("0", "false", "no", ""):
            aug_drop += 1
            continue
        p = None
        for root in roots:
            q = os.path.join(root, ip) if root else ip
            if os.path.isfile(q):
                p = q
                break
        if p is None:
            miss += 1
            continue
        # 作品级分组键: src_image_path 的所在目录（同一碑帖/作品 -> 同一组）✓
        src = (r.get("src_image_path") or "").strip()
        grp = os.path.dirname(src) if src else (r.get("source") or "unknown")
        keep.append((p, cw, sc, ch, grp))
    log("  过滤: 保留 %d | 缺图 %d | 排除集 %d | 因 aug 丢弃 %d" % (len(keep), miss, ex, aug_drop))
    return keep


def grouped_split(items, val_frac, seed, log=print):
    groups = sorted({x[4] for x in items})
    rng = random.Random(seed)
    rng.shuffle(groups)
    n_val = max(1, int(len(groups) * val_frac))
    vg = set(groups[:n_val])
    tr = [x for x in items if x[4] not in vg]
    va = [x for x in items if x[4] in vg]
    log("  作品组 %d 个 -> 验证组 %d 个 | 训练 %d / 验证 %d" % (len(groups), len(vg), len(tr), len(va)))
    return tr, va, vg


class StyleDS(Dataset):
    def __init__(self, items, w2i, s2i, size=224, train=True):
        self.it, self.w2i, self.s2i, self.size, self.train = items, w2i, s2i, size, train

    def __len__(self):
        return len(self.it)

    def __getitem__(self, i):
        p, cw, sc, _ch, _g = self.it[i]
        img = Image.open(p).convert("L").convert("RGB")
        if self.train:
            if random.random() < 0.5:
                s = random.uniform(0.95, 1.05)
                w, h = img.size
                img = img.resize((max(8, int(w * s)), max(8, int(h * s))), Image.BILINEAR)
            if random.random() < 0.4:
                w, h = img.size
                dx, dy = random.randint(-5, 5), random.randint(-5, 5)
                img = img.transform((w, h), Image.AFFINE, (1, 0, -dx, 0, 1, -dy),
                                    resample=Image.BILINEAR, fillcolor=(255, 255, 255))
            if random.random() < 0.3:
                img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 0.9)))
            if random.random() < 0.5:
                arr = np.asarray(img).astype(np.float32) * random.uniform(0.88, 1.12) + random.uniform(-15, 15)
                img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        img = img.resize((self.size, self.size), Image.BICUBIC)
        x = transforms.functional.normalize(transforms.functional.to_tensor(img),
                                           IMAGENET_MEAN, IMAGENET_STD)
        return x, self.w2i[cw], self.s2i[sc]


class StyleNet(nn.Module):
    def __init__(self, n_writer, n_script, emb=256):
        super().__init__()
        bb = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        bb.fc = nn.Identity()
        self.backbone = bb
        self.projector = nn.Sequential(nn.Linear(512, emb), nn.LayerNorm(emb), nn.ReLU())
        self.writer_head = nn.Linear(emb, n_writer)
        self.script_head = nn.Linear(emb, n_script)

    def forward(self, x):
        z = self.projector(self.backbone(x))
        return self.writer_head(z), self.script_head(z)


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    w_ok = w_tot = s_ok = s_tot = 0
    per_writer, per_script, per_script_writer = (collections.defaultdict(lambda: [0, 0]),
                                                 collections.defaultdict(lambda: [0, 0]),
                                                 collections.defaultdict(lambda: [0, 0]))
    conf = collections.defaultdict(lambda: collections.Counter())       # script -> writer 预测分布
    for x, yw, ys in loader:
        x = x.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=(dev.type == "cuda")):
            lw, ls = model(x)
        pw, ps = lw.float().argmax(1), ls.float().argmax(1)
        yw, ys = yw.to(dev), ys.to(dev)
        ok = (pw == yw)
        w_ok += int(ok.sum()); w_tot += yw.numel()
        s_ok += int((ps == ys).sum()); s_tot += ys.numel()
        for a, b, c, d in zip(yw.tolist(), ok.tolist(), ys.tolist(), pw.tolist()):
            per_writer[a][0] += int(b); per_writer[a][1] += 1
            per_script_writer[c][0] += int(b); per_script_writer[c][1] += 1
            conf[c][d] += 1
        for a, b in zip(ys.tolist(), (ps == ys).tolist()):
            per_script[a][0] += int(b); per_script[a][1] += 1
    out = {"writer_top1": w_ok / max(w_tot, 1), "script_top1": s_ok / max(s_tot, 1),
           "writer_macro": float(np.mean([c / t for c, t in per_writer.values()])) if per_writer else 0.0,
           "n": w_tot,
           "per_script_writer": {k: v[0] / v[1] for k, v in per_script_writer.items()},
           "per_script": {k: v[0] / v[1] for k, v in per_script.items()}}
    # 只看书体的多数类基线（防"看书体猜书家"）
    base, tot = 0, 0
    for sc, cnt in conf.items():
        tot += sum(cnt.values())
        base += max(cnt.values()) if cnt else 0
    out["script_only_baseline"] = base / max(tot, 1)
    return out, per_writer


def pair_test(model, items, w2i, dev, size=224, max_pairs=400, seed=0):
    """同字 × 同书体 × 异书家 的对子判别准确率（比整体 acc 更有说服力 ✓）。"""
    key = collections.defaultdict(list)
    for p, cw, sc, ch, _g in items:
        key[(ch, sc)].append((p, cw))
    rng = random.Random(seed)
    pairs = []
    for k, v in key.items():
        byw = collections.defaultdict(list)
        for p, cw in v:
            byw[cw].append(p)
        ws = [w for w in byw if len(byw[w]) > 0]
        if len(ws) >= 2:
            for _ in range(3):
                a, b = rng.sample(ws, 2)
                pairs.append((rng.choice(byw[a]), a, rng.choice(byw[b]), b))
    rng.shuffle(pairs)
    pairs = pairs[:max_pairs]
    if not pairs:
        return {"n": 0}
    model.eval()
    correct = 0
    with torch.no_grad():
        for pa, wa, pb, wb in pairs:
            ims = []
            for p in (pa, pb):
                img = Image.open(p).convert("L").convert("RGB").resize((size, size), Image.BICUBIC)
                ims.append(transforms.functional.normalize(
                    transforms.functional.to_tensor(img), IMAGENET_MEAN, IMAGENET_STD))
            x = torch.stack(ims).to(dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=(dev.type == "cuda")):
                lw, _ = model(x)
            pr = lw.float().argmax(1).tolist()
            if pr[0] == w2i[wa] and pr[1] == w2i[wb]:
                correct += 1
    return {"n": len(pairs), "both_correct": correct / len(pairs)}


@torch.no_grad()
def negative_controls(model, w2i, dev, size, font="tools/fonts/simkai.ttf"):
    model.eval()
    out = {}
    imgs = {"blank_white": Image.new("RGB", (size, size), (255, 255, 255))}
    if os.path.isfile(font):
        from PIL import ImageDraw, ImageFont
        im = Image.new("L", (512, 512), 255); f = ImageFont.truetype(font, 380)
        d = ImageDraw.Draw(im); bb = d.textbbox((0, 0), "永", font=f)
        d.text(((512 - (bb[2] - bb[0])) / 2 - bb[0], (512 - (bb[3] - bb[1])) / 2 - bb[1]),
               "永", font=f, fill=0)
        imgs["printed_永"] = im.convert("RGB")
    for name, img in imgs.items():
        x = transforms.functional.normalize(
            transforms.functional.to_tensor(img.resize((size, size), Image.BICUBIC)),
            IMAGENET_MEAN, IMAGENET_STD).unsqueeze(0).to(dev)
        with torch.autocast("cuda", dtype=torch.float16, enabled=(dev.type == "cuda")):
            lw, ls = model(x)
        pw, ps = torch.softmax(lw.float(), 1)[0], torch.softmax(ls.float(), 1)[0]
        out[name] = {"writer_max_prob": float(pw.max()), "script_max_prob": float(ps.max())}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="exp-std/csv/train.csv")
    ap.add_argument("--img-root", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--lr-backbone", type=float, default=2e-5)
    ap.add_argument("--lr-head", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--script-weight", type=float, default=0.2)
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--keep-aug", action="store_true", help="保留 aug 副本（默认丢弃 ✓）")
    ap.add_argument("--exclude-csv", default="exp-std/csv/eval200_fixed.csv")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log = lambda *x: print(*x, flush=True)

    exclude_ids = set()
    if a.exclude_csv and os.path.isfile(a.exclude_csv):
        for r in csv.DictReader(open(a.exclude_csv, encoding="utf-8-sig")):
            for k in ("img_id", "src_image_path", "image_path"):
                v = (r.get(k) or "").strip()
                if v:
                    exclude_ids.add(v); exclude_ids.add(os.path.basename(v))
        log("排除集: %d 项" % len(exclude_ids))

    items = load_rows(a.csv, a.img_root, exclude_ids, not a.keep_aug, log)
    if a.limit:
        items = items[:a.limit]
    wc = collections.Counter(x[1] for x in items)
    scc = collections.Counter(x[2] for x in items)
    log("  书家分布: %s" % wc.most_common())
    log("  书体分布: %s" % scc.most_common())
    slot = collections.Counter((x[1], x[2]) for x in items)
    tiny = [k for k, v in slot.items() if v < 30]
    log("  (书家×书体) 槽位 %d 个；样本<30 的槽位 %d 个 %s  ← 这些槽位的结果不可作为可靠辨识力 ✓"
        % (len(slot), len(tiny), tiny[:6]))
    writers = sorted(wc); scripts = sorted(scc)
    w2i = {w: i for i, w in enumerate(writers)}; s2i = {s: i for i, s in enumerate(scripts)}
    tr, va, vg = grouped_split(items, a.val_frac, a.seed, log)
    if not tr or not va:
        log("  ✗ 训练/验证为空"); return 2
    log("  writer 类=%d script 类=%d" % (len(writers), len(scripts)))

    wtc = collections.Counter(x[1] for x in tr)
    sw = torch.tensor([1.0 / math.sqrt(wtc[x[1]]) for x in tr], dtype=torch.double)
    sampler = WeightedRandomSampler(sw, num_samples=len(tr), replacement=True)
    dl_tr = DataLoader(StyleDS(tr, w2i, s2i, a.size, True), batch_size=a.batch, sampler=sampler,
                       num_workers=a.workers, pin_memory=True, drop_last=True,
                       persistent_workers=(a.workers > 0))
    dl_va = DataLoader(StyleDS(va, w2i, s2i, a.size, False), batch_size=a.batch, shuffle=False,
                       num_workers=a.workers, pin_memory=True, persistent_workers=(a.workers > 0))

    model = StyleNet(len(writers), len(scripts)).to(dev)
    head_ids = {id(p) for p in list(model.projector.parameters()) +
                list(model.writer_head.parameters()) + list(model.script_head.parameters())}
    bb = [p for p in model.parameters() if id(p) not in head_ids]
    hd = [p for p in model.parameters() if id(p) in head_ids]
    opt = torch.optim.AdamW([{"params": bb, "lr": a.lr_backbone}, {"params": hd, "lr": a.lr_head}],
                            weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs * max(1, len(dl_tr)))
    ce = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(dev.type == "cuda"))
    log("== 训练 == resnet18+双头 ~%.1fM 参数 | L_writer + %.2f*L_script | batch=%d"
        % (sum(p.numel() for p in model.parameters()) / 1e6, a.script_weight, a.batch))

    best, best_ep, hist = -1.0, -1, []
    for ep in range(1, a.epochs + 1):
        model.train(); t0 = time.time(); tot = 0; run = 0.0
        for it, (x, yw, ys) in enumerate(dl_tr):
            x, yw, ys = x.to(dev, non_blocking=True), yw.to(dev), ys.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=(dev.type == "cuda")):
                lw, ls = model(x)
                loss = ce(lw, yw) + a.script_weight * ce(ls, ys)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            run += float(loss); tot += 1
            if it % 40 == 0:
                log("   ep%d it%d loss=%.4f (%.0fs)" % (ep, it, run / max(tot, 1), time.time() - t0))
        m, per_writer = evaluate(model, dl_va, dev)
        m["d_over_baseline"] = m["writer_macro"] - m["script_only_baseline"]
        log("     [headline] writer_Macro=%.4f baseline=%.4f DELTA=%+.4f" % (m["writer_macro"], m["script_only_baseline"], m["d_over_baseline"]))
        hist.append({"epoch": ep, "loss": run / max(tot, 1), **m})
        log("  == ep%d 验证: writer Top1=%.4f Macro=%.4f | script Top1=%.4f | "
            "**只看书体基线=%.4f** | 同书体内作家=%s | 用时%.0fs"
            % (ep, m["writer_top1"], m["writer_macro"], m["script_top1"],
               m["script_only_baseline"],
               {k: round(v, 3) for k, v in m["per_script_writer"].items()}, time.time() - t0))
        if m["writer_macro"] > best:
            best, best_ep = m["writer_macro"], ep
            torch.save({"model": model.state_dict(), "writers": writers, "scripts": scripts,
                        "size": a.size, "arch": "resnet18_dual", "val_writer_macro": best,
                        "epoch": ep, "args": vars(a)}, os.path.join(a.out, "best.pt"))
            log("     ✓ 保存 best.pt (writer Macro=%.4f)" % best)
        elif ep - best_ep >= a.patience:
            log("     早停：%d 轮无提升" % a.patience); break

    _masses = [float(np.mean(1.0 - np.asarray(Image.open(x[0]).convert("L")).astype(np.float32) / 255.0)) for x in va[:400]]
    gate = {"val_ink_mean": float(np.mean(_masses)), "val_ink_p05": float(np.percentile(_masses, 5))}
    log("     ink_gate: val_mean=%.4f p05=%.4f (below p05 => near-blank, refuse to judge)" % (gate["val_ink_mean"], gate["val_ink_p05"]))
    pt = pair_test(model, va, w2i, dev, a.size, seed=a.seed)
    neg = negative_controls(model, w2i, dev, a.size)
    log("== 同字×同书体×异书家 对子判别: %s" % json.dumps(pt, ensure_ascii=False))
    log("== 负对照: %s" % json.dumps(neg, ensure_ascii=False))
    json.dump({"best_writer_macro": best, "best_epoch": best_ep, "history": hist,
               "ink_gate": gate, "pair_test": pt, "negative_controls": neg, "n_train": len(tr), "n_val": len(va),
               "writers": writers, "scripts": scripts, "args": vars(a)},
              open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    log("== 完成: writer Macro=%.4f @ep%d | 产物 %s ==" % (best, best_ep, a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
