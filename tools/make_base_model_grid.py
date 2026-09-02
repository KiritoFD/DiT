# -*- coding: utf-8 -*-
"""make_base_model_grid.py — 把不同实验在同一批 eval 样本上的生成结果拼成 grid。

用途: 直观对比生成质量演进链路
  base (s21 / s25 / s28 / s30) -> skel (s26 / s31 / 1pix) -> repa (s32b / s32c)

Grid 布局: 第 1..N 行 = 各实验生成结果, 最后一行 = GT（真迹）。列 = eval 样本 index。
每行左侧用大字号标注「实验名 + 具体做法」, 方便对比配置差异。

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

from PIL import Image, ImageDraw, ImageFont
import numpy as np

# (key, 显示名, checkpoints 目录 glob, 实验类型, 做法说明)
EXPS = [
    ("s21",  "s21 base",  "5script/results/s21_fame_flow_v2/*/checkpoints",          "base", "真迹DINO：CLS/patch注入callig+char embedding"),
    ("s25",  "s25 base",  "5script/results/s25_ids_pretrain/*/checkpoints",          "base", "IDS 部件码本替换 char embedding"),
    ("s28",  "s28 base",  "5script/results/s28_std_dino_pretrain/*/checkpoints",     "base", "标准字形DINO+PCA+OT降维（失败）"),
    ("s30",  "s30 base",  "5script/results/s30_dino_char_strong_pretrain/*/checkpoints", "base", "DINO char-strong，清洗后数据重训"),
    ("s26",  "s26 skel",  "5script/results/s26_ctrl_gt_skel/*/checkpoints",          "skel", "CtrlNet接GT骨架1px，b96（未收敛）"),
    ("s31",  "s31 skel",  "5script/results/s31_ctrl_gt_skel_1px/*/checkpoints",      "skel", "CtrlNet接GT骨架1px，b192（崩溃）"),
    ("1pix", "1pix skel", "5script/results/ctrl_fame_1pix_v1/*/checkpoints",         "skel", "ctrl_fame 1px骨架 v1（成功, SSIM 0.797）"),
    ("s32b", "s32b repa", "5script/results/s32b_repa_strong/*/checkpoints",          "repa", "repa-strong 接 1pix（SSIM 0.818）"),
    ("s32c", "s32c repa", "5script/results/s32c_chain/*/checkpoints",                "repa", "repa-chain/longconv（SSIM 0.820 最佳）"),
]

CELL = 128
PAD = 4
LABEL_W = 282
FONT_PATH = "/root/Workspace/xy/DiT/tools/fonts/SimHei.ttf"

try:
    FT_TITLE = ImageFont.truetype(FONT_PATH, 22)
    FT_DESC = ImageFont.truetype(FONT_PATH, 15)
    FT_GT = ImageFont.truetype(FONT_PATH, 22)
except Exception:
    FT_TITLE = FT_DESC = FT_GT = ImageFont.load_default()


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
    for key, label, ckpt_glob, grp, desc in sel:
        d = find_step_dir(ckpt_glob, args.step)
        if d is None:
            print(f"[skip] {key}: no step dir <= {args.step}")
            continue
        model_dirs.append((label, d, desc))
        print(f"{label}: {d}")

    if not model_dirs:
        print("no experiments")
        return

    # GT 来源: 优先第一个能找到 gt 的 base 实验
    gt_dir = None
    for _, d, _ in model_dirs:
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
        ok = all(load_gen(d, i) for _, d, _ in model_dirs)
        if ok:
            idxs.append(i)
        i += 1
    if not idxs:
        print("[err] no common samples across experiments")
        return
    print(f"samples: {idxs}")

    ncol = len(idxs)
    nrow = len(model_dirs) + 1          # +1 是底部 GT 行
    W = LABEL_W + ncol * (CELL + PAD) + PAD
    H = nrow * (CELL + PAD) + PAD
    canvas = Image.new("RGB", (W, H), (20, 22, 30))
    draw = ImageDraw.Draw(canvas)

    def put(row, col, arr):
        x = LABEL_W + col * (CELL + PAD) + PAD
        y = row * (CELL + PAD) + PAD
        canvas.paste(Image.fromarray(arr).convert("RGB"), (x, y))

    def draw_label(row, title, desc, title_font=FT_TITLE, desc_font=FT_DESC,
                   title_color=(238, 241, 248), desc_color=(172, 178, 190)):
        y_top = row * (CELL + PAD) + PAD
        x = 8
        draw.text((x, y_top + 8), title, font=title_font, fill=title_color)
        draw.text((x, y_top + 8 + 28), desc, font=desc_font, fill=desc_color)

    # 先画各实验生成行（不含 GT）
    for r, (label, d, desc) in enumerate(model_dirs):
        for c, si in enumerate(idxs):
            g = load_gen(d, si)
            put(r, c, load_gray(g))
        draw_label(r, label, desc)

    # 最后一行 = GT
    gt_row = len(model_dirs)
    for c, si in enumerate(idxs):
        put(gt_row, c, load_gray(load_gt(gt_dir, si)))
    draw_label(gt_row, "GT（真迹）", "评测集真迹，作为对照基准",
               title_font=FT_GT, desc_font=FT_DESC,
               title_color=(255, 224, 130), desc_color=(200, 180, 120))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    canvas.save(args.out)
    print(f"saved -> {args.out} ({W}x{H}) rows={[m[0] for m in model_dirs] + ['GT']}")


if __name__ == "__main__":
    main()
