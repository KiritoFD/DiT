#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""make_fame3_poster.py — fame-3 系列 eval 海报 (v10b-stdskel 布局定制).

布局: 每行 = 一个 ckpt step (黑底标签行: STEP + MSE/SSIM), 列 = n 个样本.
  行1 = STD-COND (std 标准骨架输入, 由 csv+fonts 现场渲染, 与训练条件同管线)
  行2 = GT (取最新 step 的 gt{i}.png)
  行3.. = 各 step 生成图 g{i}.png (时间升序)
数据源: <eval_root>/eval_samples_ctrl/stepXXXXXXX/g/{g,gt}{i}.png
指标: <ckpt_dir>/eval_auto_{step}.json (flat pretrain_g json)
"""
import os, re, sys, csv, glob, json, argparse, datetime
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.morphology import skeletonize

CELL, GAP = 224, 6
BG, GT_BG, GRID = (15, 17, 22), (50, 50, 50), (70, 78, 90)
HEADER_H, LABEL_H, LABEL_FONT = 40, 64, 30


def _ssim(pred, gt, win=11, data_range=1.0, sigma=1.5):
    """与 src/eval/inference._ssim 同定义 (seen 拟合分逐样本标注用)."""
    from scipy.ndimage import correlate1d
    radius = win // 2
    x_k = np.arange(-radius, radius + 1, dtype=np.float64)
    k1d = np.exp(-(x_k ** 2) / (2 * sigma ** 2))
    k1d /= k1d.sum()
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssims = []
    for ch in range(pred.shape[2]):
        x = pred[:, :, ch].astype(np.float64)
        y = gt[:, :, ch].astype(np.float64)

        def _g(img):
            img = correlate1d(img, k1d, axis=0, mode="reflect")
            return correlate1d(img, k1d, axis=1, mode="reflect")

        mx, my = _g(x), _g(y)
        mxx, myy, mxy = _g(x * x), _g(y * y), _g(x * y)
        vx, vy = mxx - mx * mx, myy - my * my
        cov = mxy - mx * my
        s = ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))
        ssims.append(s.mean())
    return float(np.mean(ssims))


def _stratify_ids(ssims, n=10):
    """按 SSIM 四分位分层选 n 个: 底 2 / 中 6 / 顶 2 (看拟合情况要覆盖卡死样本)."""
    a = np.array(ssims, dtype=np.float64)
    idx = np.argsort(a)
    qs = np.quantile(a, [0.25, 0.5, 0.75])
    buckets = [idx[a[idx] <= qs[0]],                                  # 底四分位
               idx[(a[idx] > qs[0]) & (a[idx] <= qs[1])],
               idx[(a[idx] > qs[1]) & (a[idx] <= qs[2])],
               idx[a[idx] > qs[2]]]
    take = [max(1, round(n * w)) for w in (0.2, 0.3, 0.3, 0.2)]       # 2/3/3/2
    picks = []
    for b, k in zip(buckets, take):
        if len(b) == 0:
            continue
        # 每桶内均匀取 k 个 (不含重复)
        sel = [int(b[j]) for j in np.linspace(0, len(b) - 1, min(k, len(b))).round().astype(int)]
        picks.extend(sel)
    # 补齐/截断到 n
    rest = [int(i) for i in idx if i not in picks]
    while len(picks) < n and rest:
        picks.append(rest.pop(0))
    return sorted(set(picks))[:n]

SCRIPT_FONT = {
    "楷": ["simkai.ttf", "STKAITI.TTF", "NotoSerifSC-VF.ttf"],
    "行": ["STXINGKA.TTF", "FZSTK.TTF"],
    "隶": ["SIMLI.TTF", "STLITI.TTF"],
}


def _step_key(d):
    m = re.search(r"step(\d+)", os.path.basename(d))
    return int(m.group(1)) if m else 0


def render_data/skel/std_skel(ch, script, font_dir, size=256):
    cands = list(SCRIPT_FONT.get(script, SCRIPT_FONT["楷"])) + ["simkai.ttf", "simhei.ttf"]
    seen = set()
    for f in cands:
        if f in seen:
            continue
        seen.add(f)
        p = os.path.join(font_dir, f)
        if not os.path.isfile(p):
            continue
        try:
            font = ImageFont.truetype(p, 200)
        except Exception:
            continue
        img = Image.new("L", (size, size), 255)
        d = ImageDraw.Draw(img)
        d.text((size // 2, size // 2), ch, font=font, fill=0, anchor="mm")
        a = np.asarray(img)
        if (a < 250).sum() < 10:
            continue
        sk = skeletonize(a < 127)
        arr = np.where(sk, 0, 255).astype("uint8")
        return Image.fromarray(arr).convert("RGB")
    return None


def _load_cell(path, default_bg=(40, 40, 40)):
    if path and os.path.exists(path):
        return Image.open(path).convert("RGB").resize((CELL, CELL), Image.LANCZOS)
    return Image.new("RGB", (CELL, CELL), default_bg)


def _fonts():
    font = font_s = font_lab = None
    for _fp, _sz in [(r"C:\Windows\Fonts\msyh.ttc", 17),
                     ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17),
                     (r"C:\Windows\Fonts\msyhbd.ttc", LABEL_FONT),
                     ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", LABEL_FONT)]:
        try:
            _f = ImageFont.truetype(_fp, _sz)
            if font is None: font = _f
            elif font_s is None: font_s = _f
            elif font_lab is None: font_lab = _f
        except Exception:
            pass
    if font is None:
        font = font_s = ImageFont.load_default()
    if font_lab is None:
        font_lab = font
    return font, font_s, font_lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True,
                    help="含 eval_auto_*.json 的目录 (checkpoints)")
    ap.add_argument("--sample-root", default=None,
                    help="eval_samples_ctrl 根目录 (默认 ckpt-dir 同级)")
    ap.add_argument("--json-dir", default=None,
                    help="eval_auto_*.json 目录 (默认 = ckpt-dir)")
    ap.add_argument("--csv", default=None,
                    help="eval 样本对应 csv (默认回退 data_csv 前 n 行, 供 STD-COND 渲染)")
    ap.add_argument("--font-dir", default="/root/Workspace/xy/DiT/tools/fonts")
    ap.add_argument("--exp", default="")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--ids", default="",
                    help="逗号分隔样本下标 (seen eval 的样本序号 = data_csv 行号); 空=前 n 个")
    ap.add_argument("--stratify", action="store_true",
                    help="按参照 step 逐样本 SSIM 四分位分层选 n 个 (底2/中6/顶2), 列头标注逐样本 SSIM")
    ap.add_argument("--step-stride", type=int, default=1)
    ap.add_argument("-o", "--out", default="/tmp/fame3_poster.png")
    args = ap.parse_args()

    json_dir = args.json_dir or args.ckpt_dir
    arm_root = args.sample_root or os.path.join(os.path.dirname(args.ckpt_dir.rstrip("/")), "eval_samples_ctrl")
    step_dirs = {(_step_key(d)): d for d in glob.glob(os.path.join(arm_root, "step*"))}
    steps_all = sorted(step_dirs)
    steps = [s for s in steps_all[::args.step_stride]
             if os.path.exists(os.path.join(step_dirs[s], "g", "g0.png"))
             and os.path.exists(os.path.join(step_dirs[s], "g", "gt0.png"))]
    if not steps:
        print("[poster] no step dirs"); return 1

    # 参照 step: 最近一个已画完的目录 (g0 存在且 gt0 存在)
    ref = None
    for s in reversed(steps_all):
        d0 = step_dirs[s]
        if os.path.exists(os.path.join(d0, "g", "g0.png")) and \
           os.path.exists(os.path.join(d0, "g", "gt0.png")):
            ref = d0
            break
    if ref is None:
        print("[poster] no completed step dir yet"); return 1
    latest = ref
    n_all = 0
    for i in range(100):
        if os.path.exists(os.path.join(latest, "g", f"g{i}.png")):
            n_all = i + 1
        else:
            break

    # 样本选择: --ids 显式 / --stratify 按参照 step SSIM 四分位分层 / 默认前 n
    ref_ssims = None
    if args.stratify:
        from PIL import Image as _I
        ref_ssims = []
        for i in range(n_all):
            p = os.path.join(latest, "g", f"g{i}.png")
            q = os.path.join(latest, "g", f"gt{i}.png")
            if not (os.path.exists(p) and os.path.exists(q)):
                ref_ssims.append(np.nan)
                continue
            pred = np.asarray(_I.open(p).convert("RGB"), np.float32) / 255.0
            gt = np.asarray(_I.open(q).convert("RGB"), np.float32) / 255.0
            ref_ssims.append(_ssim(pred, gt))
        ref_ssims = np.array(ref_ssims)
    if args.ids:
        ids = [int(x) for x in args.ids.split(",") if x != ""]
    elif args.stratify:
        ids = _stratify_ids(ref_ssims, n=args.n)
    else:
        ids = list(range(min(n_all, args.n)))
    n = len(ids)
    print(f"[poster] steps={[s for s in steps]}, n_samples={n}, ids={ids}")

    # 指标
    ev_map = {}
    for f in glob.glob(os.path.join(json_dir, "eval_auto_*.json")):
        try:
            d = json.load(open(f))
            ev_map[int(d.get("step", 0))] = (d.get("mse"), d.get("ssim"), d.get("skel_iou"))
        except Exception:
            pass

    # csv 行 (STD-COND 行用)
    csv_path = args.csv
    if csv_path is None:
        try:
            import torch
            a = torch.load(os.path.join(json_dir, f"{steps_all[-1]:07d}.pt"),
                           map_location="cpu", weights_only=False).get("args")
            a = vars(a) if isinstance(a, argparse.Namespace) else (a or {})
            csv_path = a.get("gpu_eval_csv") or a.get("eval_csv") or a.get("data_csv")
            if csv_path and not os.path.isabs(csv_path):
                csv_path = os.path.join("/root/Workspace/xy/DiT", csv_path)
        except Exception:
            pass
    rows_all = []
    if csv_path and os.path.exists(csv_path):
        rows_all = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    rows = [rows_all[i] for i in ids if i < len(rows_all)]
    print(f"[poster] csv={csv_path} rows={len(rows)}")

    W = CELL * n + GAP * 2
    n_rows = 2 + len(steps)   # STD-COND + GT + per-step
    H = GAP + HEADER_H + n_rows * (LABEL_H + CELL + GAP) + GAP + 30
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)
    font, font_s, font_lab = _fonts()

    def label_row(y, text, color=(255, 255, 255)):
        draw.rectangle([0, y, W, y + LABEL_H], fill=(0, 0, 0))
        draw.line([(0, y + LABEL_H - 1), (W, y + LABEL_H - 1)], fill=GRID)
        draw.text((12, y + (LABEL_H - LABEL_FONT) // 2), text, font=font_lab, fill=color)

    y = GAP
    # 顶部
    if ref_ssims is not None:
        chars = " | ".join(
            f"{(rows_all[i].get('character', '') or '?') if i < len(rows_all) else '?'} "
            f"{ref_ssims[i]:.2f}" if not np.isnan(ref_ssims[i]) else
            f"{(rows_all[i].get('character', '') or '?') if i < len(rows_all) else '?'}  n/a"
            for i in ids) or "n/a"
    else:
        chars = " | ".join((r.get("character", "") or "?") for r in rows) or "n/a"
    draw.text((GAP + 6, y + 2), "rows: STD-COND (输入骨架) / GT / 各 step 生成", font=font, fill=(120, 200, 255))
    draw.text((GAP + 6, y + 20), f"cols: {chars}", font=font_s, fill=(150, 165, 190))
    if args.exp:
        draw.text((W - 6, 4), args.exp, anchor="ra", font=font_s, fill=(90, 100, 120))
    y += HEADER_H

    # STD-COND 行
    label_row(y, "STD-COND  (标准骨架输入)", color=(120, 200, 255)); y += LABEL_H
    x = GAP
    for r in rows:
        img = render_data/skel/std_skel(r.get("character", ""), r.get("script", "楷"), args.font_dir)
        canvas.paste(_load_cell(None) if img is None else img.resize((CELL, CELL), Image.LANCZOS), (x, y))
        x += CELL
    y += CELL + GAP

    # GT 行
    label_row(y, "GT (真值)"); y += LABEL_H
    x = GAP
    for i in ids:
        canvas.paste(_load_cell(os.path.join(latest, "g", f"gt{i}.png"), GT_BG), (x, y))
        x += CELL
    y += CELL + GAP

    # 各 step 行
    for step in steps:
        d = step_dirs[step]
        parts = [f"STEP {step}"]
        m = ev_map.get(step)
        if m:
            if m[0] is not None: parts.append("MSE %.3f" % m[0])
            if m[1] is not None: parts.append("SSIM %.3f" % m[1])
        label_row(y, "    ".join(parts)); y += LABEL_H
        x = GAP
        for i in ids:
            canvas.paste(_load_cell(os.path.join(d, "g", f"g{i}.png")), (x, y))
            x += CELL
        y += CELL + GAP

    draw.text((6, y + 4),
              f"{len(steps)} ckpt x {n} samples | g=std-skel cond, cfg 0.7 | "
              f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
              font=font_s, fill=(90, 100, 120))
    canvas.save(args.out)
    print(f"[poster] -> {args.out} ({W}x{H})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
