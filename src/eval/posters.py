# -*- coding: utf-8 -*-
"""eval poster 生成 (v10b-stdskel review 布局, 沿袭 _ot_scratch/posters_200k.py).

- seen : 全部样本 (默认 n=10) 拼 1 张
- strict: 前 N 个样本 (默认 50) 每张拼 per_poster 个 (默认 10) -> 50/10 = 5 张
- 每格: 左=模型输出(绿/红框) 右=GT, 下方标 字/书家/ssim
输出: <run_dir>/posters/step{step:07d}_{seen|strict_p{k}}.png
"""
import argparse
import csv
import glob
import json
import os
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import uniform_filter

CELL, PAIR_GAP, ROW_GAP = 140, 14, 34
COLS = 5
BG = "white"


def _font_path():
    for p in ("/root/Workspace/xy/DiT/tools/fonts/simhei.ttf", "/tmp/simhei.ttf",
              "tools/fonts/simhei.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.isfile(p):
            return p
    return None


def _ssim(a, b, win=7):
    """a,b: (H,W,3) float 0..1 -> 逐通道 SSIM 均值。

    ★ 2026-09-17: 实现收束到 src/eval/metrics.py（原 docstring 写"与推理评测同定义"
    是**错的** —— 主评测路径用的是高斯窗 win=11，这里是均匀窗 win=7，数值差约 0.001）。
    现在用 `window="box"` 显式声明本处口径，**数值不变**，但实现只剩一份。
    若要与其他地方对比，请确认窗口口径一致。
    """
    from src.eval.metrics import ssim as _ssim_impl
    return _ssim_impl(a, b, win=win, window="box")


def _load_rgb(path, cell):
    if not os.path.exists(path):
        return None
    return Image.open(path).convert("RGB").resize((cell, cell), Image.LANCZOS)


def _load_np(path):
    if not os.path.exists(path):
        return None
    return np.asarray(Image.open(path).convert("RGB").resize((112, 112)),
                      dtype=np.float32) / 255.0


def _csv_rows(path, n):
    p = path if os.path.isabs(path) else os.path.join("/root/Workspace/xy/DiT", path)
    if not os.path.exists(p):
        return []
    return list(csv.DictReader(open(p, encoding="utf-8")))[:n]


def _make_sheet(items, title, out_path, font, font_s, cell=CELL):
    """items: [(idx_label, char, calligrapher, ssim, model_path, gt_path), ...]"""
    n = len(items)
    cols = min(COLS, n) if n > 0 else 1
    rows = (n + cols - 1) // cols
    w = 10 + cols * (cell * 2 + PAIR_GAP) + 10
    h = 50 + rows * (cell + ROW_GAP) + 20
    cv = Image.new("RGB", (w, h), BG)
    dr = ImageDraw.Draw(cv)
    dr.text((12, 8), title, font=font, fill="black")
    for i, (lab, char, callig, s, pm, gm) in enumerate(items):
        rr, cc = divmod(i, cols)
        x = 10 + cc * (cell * 2 + PAIR_GAP)
        y = 50 + rr * (cell + ROW_GAP)
        po = _load_rgb(pm, cell)
        pg = _load_rgb(gm, cell)
        if po is not None:
            cv.paste(po, (x, y))
        if pg is not None:
            cv.paste(pg, (x + cell + 8, y))
        bad = (s is not None and s < 0.45)
        dr.rectangle([x, y, x + cell - 1, y + cell - 1],
                     outline="#CC0000" if bad else "#00AA00", width=2)
        dr.rectangle([x + cell + 8, y, x + cell * 2 + 7, y + cell - 1],
                     outline="#CCCCCC")
        s_txt = f"ssim={s:.2f}" if s is not None else "ssim=n/a"
        dr.text((x, y + cell + 3), f"{lab} {char} {callig} {s_txt}",
                font=font_s, fill="#CC0000" if bad else "black")
    cv.save(out_path)
    return out_path


def make_posters(run_dir, step, out_dir=None, n_seen=10, n_strict=50,
                 strict_per=10, which=("seen", "strict"),
                 seen_csv="assets/eval_seen_v10.csv",
                 strict_csv="assets/eval_fame3_strict_clean_v9.csv",
                 tag="", prefix=""):
    """为一个 step 生成 poster; 返回生成的文件路径列表."""
    step_dir = os.path.join(run_dir, "eval_samples_ctrl", f"step{step:07d}")
    out_dir = out_dir or os.path.join(run_dir, "posters")
    os.makedirs(out_dir, exist_ok=True)
    pre = f"{prefix}_" if prefix else ""
    fp = _font_path()
    font = ImageFont.truetype(fp, 22) if fp else ImageFont.load_default()
    font_s = ImageFont.truetype(fp, 14) if fp else ImageFont.load_default()
    made = []

    if "seen" in which:
        d = os.path.join(step_dir, "g")
        imgs = sorted(glob.glob(os.path.join(d, "g[0-9]*.png")),
                      key=lambda p: int(re.search(r"g(\d+)\.png", p).group(1)))
        rows = _csv_rows(seen_csv, max(len(imgs), n_seen))
        items = []
        for i in range(min(n_seen, len(imgs))):
            pm, gm = os.path.join(d, f"g{i}.png"), os.path.join(d, f"gt{i}.png")
            a, b = _load_np(pm), _load_np(gm)
            s = _ssim(a, b) if (a is not None and b is not None) else None
            r = rows[i] if i < len(rows) else {}
            items.append((f"#{i}", r.get("character", "?"), r.get("calligrapher", ""),
                          s, pm, gm))
        if items:
            out = os.path.join(out_dir, f"{pre}step{step:07d}_seen.png")
            made.append(_make_sheet(
                items, f"{tag} {step/1000:.1f}k SEEN{len(items)} — 左=模型 右=GT", out,
                font, font_s))

    if "strict" in which:
        d = os.path.join(step_dir, "strict50")
        if not os.path.exists(os.path.join(d, "g0.png")):
            d = os.path.join(step_dir, "strict")
        rows = _csv_rows(strict_csv, n_strict)
        n_avail = min(n_strict, sum(1 for i in range(n_strict)
                                    if os.path.exists(os.path.join(d, f"g{i}.png"))))
        n_part = max(1, (n_avail + strict_per - 1) // strict_per) if n_avail else 0
        for k in range(n_part):
            items = []
            for i in range(k * strict_per, min((k + 1) * strict_per, n_avail)):
                pm, gm = os.path.join(d, f"g{i}.png"), os.path.join(d, f"gt{i}.png")
                a, b = _load_np(pm), _load_np(gm)
                s = _ssim(a, b) if (a is not None and b is not None) else None
                r = rows[i] if i < len(rows) else {}
                items.append((f"#{i}", r.get("character", "?"),
                              r.get("calligrapher", ""), s, pm, gm))
            out = os.path.join(out_dir, f"{pre}step{step:07d}_strict_p{k}.png")
            made.append(_make_sheet(
                items,
                f"{tag} {step/1000:.1f}k STRICT{n_avail} [{k+1}/{n_part}] "
                f"#{k*strict_per}-#{min((k+1)*strict_per, n_avail)-1} — 左=模型 右=GT",
                out, font, font_s))
    return made


def _latest_step_dirs(run_dir):
    steps = []
    for d in glob.glob(os.path.join(run_dir, "eval_samples_ctrl", "step*")):
        m = re.search(r"step(\d+)", os.path.basename(d))
        if m:
            steps.append(int(m.group(1)))
    return sorted(steps)


def _step_metrics(run_dir, step, set_name):
    """ckpt 的 eval_auto_{step}.json -> (ssim, ) 取 seen 平铺或 strict 嵌套."""
    p = os.path.join(run_dir, "checkpoints", f"eval_auto_{step}.json")
    if not os.path.exists(p):
        return None
    try:
        d = json.load(open(p, encoding="utf-8"))
    except Exception:
        return None
    if set_name == "strict":
        st = d.get("strict") or {}
        return st.get("ssim_mean")
    return d.get("ssim")


def make_aggregate(run_dir, set_name, out_path=None, cell=None,
                   n_seen=10, n_strict=50,
                   seen_csv="assets/eval_seen_v10.csv",
                   strict_csv="assets/eval_fame3_strict_clean_v9.csv",
                   tag="", prefix=""):
    """总集 poster: 每行一个 ckpt, 末行 GT; 列 = 样本.

    seen : 10 样本; strict: 50 样本. 输出 <run_dir>/posters/aggregate_{set}.png
    """
    sub = "g" if set_name == "seen" else "strict"
    n_samples = n_seen if set_name == "seen" else n_strict
    csv_p = seen_csv if set_name == "seen" else strict_csv
    rows_csv = _csv_rows(csv_p, n_samples)

    # 哪些 step 有该集样本 (至少 g0)
    steps_all = _latest_step_dirs(run_dir)
    steps = []
    for st in steps_all:
        d = os.path.join(run_dir, "eval_samples_ctrl", f"step{st:07d}", sub)
        if os.path.exists(os.path.join(d, "g0.png")):
            steps.append(st)
    if not steps:
        return None

    # 列数 = 实际可用样本数 (以最新 step 为准)
    d_last = os.path.join(run_dir, "eval_samples_ctrl", f"step{steps[-1]:07d}", sub)
    n_avail = 0
    for i in range(n_samples):
        if os.path.exists(os.path.join(d_last, f"g{i}.png")):
            n_avail = i + 1
        else:
            break
    if n_avail == 0:
        return None

    cell = cell or (148 if set_name == "seen" else 96)
    lab_w = 118
    gap = 3
    head_h = 46
    row_h = cell + 8
    n_rows = len(steps) + 1        # + GT
    w = lab_w + n_avail * (cell + gap) + 12
    h = head_h + n_rows * row_h + 24
    cv = Image.new("RGB", (w, h), BG)
    dr = ImageDraw.Draw(cv)
    fp = _font_path()
    font = ImageFont.truetype(fp, 22) if fp else ImageFont.load_default()
    font_s = ImageFont.truetype(fp, max(11, cell // 9)) if fp else ImageFont.load_default()
    font_l = ImageFont.truetype(fp, 13) if fp else ImageFont.load_default()
    title = (f"{tag} {set_name.upper()}{n_avail} 总集 — 每行=ckpt (左标 step/ssim), 末行=GT")
    dr.text((10, 8), title, font=font, fill="black")

    # 行标签 + 图
    for r, st in enumerate(steps + ["GT"]):
        y = head_h + r * row_h
        if st == "GT":
            dr.text((8, y + cell // 2 - 8), "GT", font=font, fill="black")
            src_dir = os.path.join(run_dir, "eval_samples_ctrl",
                                   f"step{steps[-1]:07d}", sub)
            for i in range(n_avail):
                img = _load_rgb(os.path.join(src_dir, f"gt{i}.png"), cell)
                if img is not None:
                    cv.paste(img, (lab_w + i * (cell + gap), y))
            continue
        s = _step_metrics(run_dir, st, set_name)
        lab1 = f"{st/1000:.1f}k"
        lab2 = (f"ssim {s:.3f}" if s is not None else "ssim --")
        dr.text((8, y + cell // 2 - 16), lab1, font=font_l, fill="black")
        dr.text((8, y + cell // 2 + 2), lab2, font=font_l, fill="#555555")
        src_dir = os.path.join(run_dir, "eval_samples_ctrl", f"step{st:07d}", sub)
        for i in range(n_avail):
            img = _load_rgb(os.path.join(src_dir, f"g{i}.png"), cell)
            if img is not None:
                cv.paste(img, (lab_w + i * (cell + gap), y))
    _pre = f"{prefix}_" if prefix else ""
    out_path = out_path or os.path.join(run_dir, "posters",
                                        f"{_pre}aggregate_{set_name}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv.save(out_path)
    return out_path



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="实验 run 目录 (含 eval_samples_ctrl)")
    ap.add_argument("--step", default="latest", help="'latest' | 'all' | 数字")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--n-seen", type=int, default=10)
    ap.add_argument("--n-strict", type=int, default=50)
    ap.add_argument("--strict-per", type=int, default=10)
    ap.add_argument("--sets", default="seen,strict")
    ap.add_argument("--agg-sets", default="seen,strict",
                    help="生成总集 poster 的集合 (空=不生成)")
    ap.add_argument("--agg-seen-cell", type=int, default=148)
    ap.add_argument("--agg-strict-cell", type=int, default=96)
    ap.add_argument("--seen-csv", default="assets/eval_seen_v10.csv")
    ap.add_argument("--strict-csv", default="assets/eval_fame3_strict_clean_v9.csv")
    ap.add_argument("--tag", default="")
    ap.add_argument("--prefix", default="", help="输出文件名前缀 (避免跨实验撞名)")
    args = ap.parse_args()

    steps = _latest_step_dirs(args.run_dir)
    if not steps:
        print("[posters] no step dirs"); return 1
    if args.step == "latest":
        todo = [steps[-1]]
    elif args.step == "all":
        todo = steps
    else:
        todo = [int(args.step)]
    which = tuple(s for s in args.sets.split(",") if s)
    for st in todo:
        made = make_posters(args.run_dir, st, args.out_dir, args.n_seen,
                            args.n_strict, args.strict_per, which,
                            seen_csv=args.seen_csv, strict_csv=args.strict_csv,
                            tag=args.tag, prefix=args.prefix)
        for p in made:
            print(f"[posters] {p}")
    for s in (x for x in args.agg_sets.split(",") if x):
        cell = args.agg_seen_cell if s == "seen" else args.agg_strict_cell
        p = make_aggregate(args.run_dir, s, cell=cell, tag=args.tag,
                           prefix=args.prefix,
                           n_seen=args.n_seen, n_strict=args.n_strict,
                           seen_csv=args.seen_csv, strict_csv=args.strict_csv)
        if p:
            print(f"[posters] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
