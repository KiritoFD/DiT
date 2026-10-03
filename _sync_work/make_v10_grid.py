# -*- coding: utf-8 -*-
"""make_v10_grid.py — v10 三臂 **best ckpt** 推理结果对比 grid (远程跑).

改版要点:
  1) 每臂按 eval json 的 **SSIM 最高** 选 best ckpt（不再用"最新 step"）
  2) 组别文字放大（字号 ~40，含 臂名 / step / SSIM）

用法(远程): /opt/conda/bin/python _sync_work/make_v10_grid.py
产物: /root/Workspace/xy/DiT/_diag/v10_grid.png
"""
import os
import re
import glob
import json
from PIL import Image, ImageDraw, ImageFont

ROOT = "/root/Workspace/xy/DiT/assets/results"
OUT = "/root/Workspace/xy/DiT/_diag/v10_grid.png"
N_SAMPLE = 12
CELL = 256
FONT_SIZE = 40
LABEL_H = 78          # 每行标签区高度

ARMS = [
    ("v10a  (random char tbl)", "v10a_skel_cond_pretrain"),
    ("v10a-dino  (frozen DINO)", "v10adino_skel_cond_pretrain"),
    ("v10b  (no char cond)", "v10b_skel_only_pretrain"),
]


def get_font(size):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if os.path.isfile(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def pick_best(model_dir):
    """SSIM 最高的 eval json -> (ssim, step)。"""
    pats = glob.glob(os.path.join(model_dir, "*", "checkpoints", "eval_auto_*.json"))
    pats += glob.glob(os.path.join(model_dir, "checkpoints", "eval_auto_*.json"))
    best = None
    for j in pats:
        try:
            d = json.load(open(j, encoding="utf-8"))
        except Exception:
            continue
        s = d.get("ssim")
        st = d.get("step")
        if s is None or st is None:
            continue
        if best is None or s > best[0]:
            best = (float(s), int(st))
    return best


def find_step_dir(model_dir, step):
    """定位该 step 的样本目录；不存在则回退到有样本的最新 step。"""
    pats = glob.glob(os.path.join(model_dir, "*", "eval_samples_ctrl", "step*"))
    if not pats:
        pats = glob.glob(os.path.join(model_dir, "eval_samples_ctrl", "step*"))

    def num(p):
        m = re.search(r"(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else -1

    def has_sample(p):
        gd = os.path.join(p, "g")
        if not os.path.isdir(gd):
            return False
        return os.path.isfile(os.path.join(gd, "g0.png"))

    want = [p for p in pats if num(p) == step and has_sample(p)]
    if want:
        return want[0], False
    cands = [p for p in pats if has_sample(p)]
    if not cands:
        return None, False
    return max(cands, key=num), True


def main():
    font = get_font(FONT_SIZE)
    font_sm = get_font(int(FONT_SIZE * 0.7))
    rows = []
    gt_row = None
    for label, mdir in ARMS:
        d = os.path.join(ROOT, mdir)
        best = pick_best(d)
        if best is None:
            print(f"[skip] {label}: no eval json")
            continue
        ssim, step = best
        step_dir, fell = find_step_dir(d, step)
        if step_dir is None:
            print(f"[skip] {label}: no samples")
            continue
        imgs, gts = [], []
        for i in range(N_SAMPLE):
            g = os.path.join(step_dir, "g", f"g{i}.png")
            gt = os.path.join(step_dir, "g", f"gt{i}.png")
            if not (os.path.isfile(g) and os.path.isfile(gt)):
                continue
            imgs.append(Image.open(g).convert("L").resize((CELL, CELL)))
            gts.append(Image.open(gt).convert("L").resize((CELL, CELL)))
        if not imgs:
            print(f"[skip] {label}: empty samples")
            continue
        if gt_row is None:
            gt_row = gts
        tag = f"SSIM {ssim:.4f}  @step {step}"
        if fell:
            tag += "  (samples: latest avail)"
        rows.append((label, tag, imgs))
        print(f"[ok] {label}: {len(imgs)} samples  {tag}")

    if not rows:
        print("no data")
        return

    ncol = len(rows[0][2])
    W = 30 + CELL * ncol
    H = 24 + (LABEL_H + CELL) * (len(rows) + 1)
    canvas = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(canvas)

    dr.text((14, 8), "GT  (ground truth)", font=font, fill="black")
    for j, im in enumerate(gt_row):
        canvas.paste(im.convert("RGB"), (30 + j * CELL, 14 + LABEL_H))

    for r, (label, tag, imgs) in enumerate(rows):
        y0 = 14 + (LABEL_H + CELL) * (r + 1)
        dr.text((14, y0 + 6), label, font=font, fill="#1a4fd0")
        dr.text((14, y0 + 6 + FONT_SIZE + 4), tag, font=font_sm, fill="#666666")
        for j, im in enumerate(imgs):
            canvas.paste(im.convert("RGB"), (30 + j * CELL, y0 + LABEL_H))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    canvas.save(OUT)
    print(f"[saved] {OUT} size={canvas.size}")


if __name__ == "__main__":
    main()
