# -*- coding: utf-8 -*-
"""make_base_model_grid.py — 把不同实验在同一批 eval 样本上的生成结果拼成 grid。

用途: 直观对比生成质量演进链路
  base (s21 / s25 / s28 / s30) -> skel (s26 / s31 / 1pix) -> repa (s32b / s32c)

Grid 布局: 第 1 行 = GT, 之后每行 = 一个实验的生成结果。列 = eval 样本 index。

不同实验的 eval_samples 目录结构不一致, loader 自动探测:
  - base 实验: eval_samples/stepXXXX/sample{i}.png  + gt{i}.png
  - skel/repa : eval_samples_ctrl/stepXXXX/ctrl/ctrl{i}.png (+ base/base{i}.png)
GT 行取自任一能找到 gt 的 base 实验 (假设各实验用同一份 eval 集顺序)。

用法:
  python tools/make_base_model_grid.py --group all --step 30000 --n 8 --out _sync_work/grid_all.png
  python tools/make_base_model_grid.py --group base --step 30000 --n 8
"""
import os
import sys
import glob
import argparse

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir("/root/Workspace/xy/DiT")

from PIL import Image, ImageDraw
import numpy as np

# (key, 显示名, checkpoints 目录 glob, 实验类型)
EXPS = [
    ("s21", "s21 base",        "5script/results/s21_fame_flow_v2/*/checkpoints", "base"),
    ("s25", "s25 base",        "5script/results/s25_ids_pretrain/*/checkpoints", "base"),
    ("s28", "s28 base",        "5script/results/s28_std_dino_pretrain/*/checkpoints", "base"),
    ("s30", "s30 base",        "5script/results/s30_dino_char_strong_pretrain/*/checkpoints", "base"),
    ("s26", "s26 skel",        "5script/results/s26_ctrl_gt_skel/*/checkpoints", "skel"),
    ("s31", "s31 skel",        "5script/results/s31_ctrl_gt_skel_1px/*/checkpoints", "skel"),
    ("1pix", "1pix skel",      "5script/results/ctrl_fame_1pix_v1/*/checkpoints", "skel"),
    ("s32b", "s32b repa",      "5script/results/s32b_repa_strong/*/checkpoints", "repa"),
    ("s32c", "s32c repa",      "5script/results/s32c_chain/*/checkpoints", "repa"),
]

CELL = 128
PAD = 4
LABEL_W = 78


def find_step_dir(ckpt_glob, step, prefer_ctrl=True):
    """在 checkpoints 目录下找 stepXXXX 子目录 (支持 eval_samples 与 eval_samples_ctrl)。"""
    cands = []
    for ck in glob.glob(ckpt_glob):
        for sub in ("eval_samples", "eval_samples_ctrl"):
            cands += glob.glob(os.path.join(ck, sub, "step*"))
    if not cands:
        return None
    best, best_s = None, -1
    for d in cands:
        try:
            s = int(os.path.basename(d).replace("step", ""))
        except ValueError:
            continue
        if s <= step and s > best_s:
            best, best_s = d, s
    return best if best is not None else (sorted(cands)[-1] if cands else None)


def _try(stepdir, patterns):
    for p in patterns:
        f = os.path.join(stepdir, p)
        if os.path.isfile(f):
            return f
    return None


def load_gen(stepdir, i):
    return _try(stepdir, [
        f"sample{i}.png",
        f"ctrl/ctrl{i}.png",
        f"base/base{i}.png",
    ])


def load_gt(stepdir, i):
    return _try(stepdir, [
        f"gt{i}.png",
        f"ctrl/gt{i}.png",
        f"base/gt{i}.png",
    ])


def load_gray(path, size=(CELL, CELL)):
    try:
        img = Image.open(path).convert("L").resize(size, Image.LANCZOS)
    except Exception:
        img = Image.new("L", size, 255)
    return np.asarray(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=30000)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--group", default="all", choices=["all", "base", "skel", "repa"])
    ap.add_argument("--out", default="_sync_work/model_grid.png")
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()

    sel = [e for e in EXPS if args.group == "all" or e[3] == args.group]
    model_dirs = []
    for key, label, ckpt_glob, grp in sel:
        d = find_step_dir(ckpt_glob, args.step)
        if d is None:
            print(f"[skip] {key}: no step dir <= {args.step}")
            continue
        model_dirs.append((label, d))
        print(f"{label}: {d}")

    if not model_dirs:
        print("no experiments")
        return

    # GT 来源: 优先第一个能找到 gt 的 base 实验
    gt_dir = None
    for _, d in model_dirs:
        if load_gt(d, 0):
            gt_dir = d
            break
    if gt_dir is None:
        gt_dir = model_dirs[0][1]
    print(f"GT source: {gt_dir}")

    # 对齐: 找所有实验 gen 都存在 + gt 存在的 index
    idxs = []
    i = args.start
    scanned = 0
    while len(idxs) < args.n and scanned < 600:
        scanned += 1
        gt = load_gt(gt_dir, i)
        if gt is None:
            i += 1
            continue
        ok = all(load_gen(d, i) for _, d in model_dirs)
        if ok:
            idxs.append(i)
        i += 1
    if not idxs:
        print("[err] no common samples across experiments")
        return
    print(f"samples: {idxs}")

    ncol, nrow = len(idxs), 1 + len(model_dirs)
    W = LABEL_W + ncol * (CELL + PAD) + PAD
    H = nrow * (CELL + PAD) + PAD
    canvas = Image.new("RGB", (W, H), (20, 22, 30))
    draw = ImageDraw.Draw(canvas)

    def put(row, col, arr):
        x = LABEL_W + col * (CELL + PAD) + PAD
        y = row * (CELL + PAD) + PAD
        canvas.paste(Image.fromarray(arr).convert("RGB"), (x, y))

    for c, si in enumerate(idxs):
        put(0, c, load_gray(load_gt(gt_dir, si)))
    draw.text((6, PAD + CELL // 2), "GT", fill=(230, 233, 239))

    for r, (label, d) in enumerate(model_dirs, start=1):
        for c, si in enumerate(idxs):
            g = load_gen(d, si)
            put(r, c, load_gray(g))
        draw.text((6, r * (CELL + PAD) + PAD + CELL // 2), label, fill=(230, 233, 239))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    canvas.save(args.out)
    print(f"saved -> {args.out} ({W}x{H}) rows={[m[0] for m in model_dirs]}")


if __name__ == "__main__":
    main()
