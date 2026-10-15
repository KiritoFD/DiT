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

# ══════════ 在训海报（2026-10-14 [重构 PR5] 从 in_mem_eval.py 搬来 ✓）══════════
def _poster_canny(img):
    """从 gen 图现算 canny 边缘列。

    ⚠ 极性: 统一成**白底黑线**, 与数据侧约定一致。
    库原生输出 (cv2.Canny / 梯度阈值) 是"前景=255/背景=0"(黑底白线), 直接落盘
    会与 final_canny_base / skel3 PNG 的极性相反 -> 看起来像"数据错了"。
    这与 2026-09-14 的 E1(canny 黑底未归一)属同一类疏漏: 别把库的原生输出直接当交付格式。
    """
    import cv2
    a = np.asarray(img, dtype=np.float32)
    gray = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    gx = cv2.filter2D(gray, -1, kx, borderType=cv2.BORDER_REFLECT)
    gy = cv2.filter2D(gray, -1, ky, borderType=cv2.BORDER_REFLECT)
    edge = np.sqrt(gx ** 2 + gy ** 2) > 150
    return Image.fromarray(np.where(edge, 0, 255).astype(np.uint8)).convert("RGB")

def _poster_skeleton(img):
    """从 gen 图现算骨架列。

    ⚠ 极性同 `_poster_canny`: 输出**白底黑线**(骨架=0), 而不是 skeletonize 的
    原生 "骨架=255" —— 否则与 skel3 PNG / 输入 g 的极性相反, 造成误读。
    """
    from skimage.morphology import skeletonize
    a = np.asarray(img, dtype=np.float32)
    gray = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    bin_b = (gray > 127).astype(np.uint8)
    if gray.mean() > 127:
        bin_b = 1 - bin_b
    sk = skeletonize(bin_b.astype(bool))
    return Image.fromarray(np.where(sk, 0, 255).astype(np.uint8)).convert("RGB")

def render_poster(results_dir, set_name, out=None, cell=224, gap=6,
                  train_csv=None, eval_csv=None, n_eval=None):
    """自动 poster (每 set 两张):

    主图 posters/{set}_poster.png:
      第 1 行 = 输入标准字 (eval_samples_ctrl/{set}_input_g/g{i}.png)
      中间每行 = 一个 ckpt 的生成结果 (时间升序, 行标签带 ssim)
      最后 1 行 = GT (最新 step 的 gt{i}.png)
    结构图 posters/{set}_struct.png (另放, 不挤主图):
      每 ckpt 一行, 每样本 gen | canny | skel 三列。
    均为全量重画并覆盖 → 永远是所有 step 的最新版。纯 CPU, 秒级。
    """
    # ⚠ 临时兼容（2026-10-14 [重构 PR5]）: 这 6 个助手仍在 in_mem_eval.py ✓
    #   ⇒ 用**函数内导入**打破循环（posters ← in_mem_eval ← posters ✗）✓
    #   ⇒ 后续把它们也搬走即可删掉本段 ✓
    from src.eval.in_mem_eval import (  # noqa: E402
        _steps_scan, _load_font, _font_cache, _train_ref_rows, _train_nn_row, _step_ssim_txt)
    from PIL import Image as _Img, ImageDraw
    sub = "g" if set_name in ("seen", "g") else set_name
    steps = _steps_scan(results_dir, sub)
    if not steps:
        return None
    n_max = max(s[2] for s in steps)
    # ★ 2026-10-01 修复: 原实现 cell = 1280//n_max (20 列 -> 64px) 且用 LANCZOS 缩小,
    #   1~2px 的骨架细线被邻域平均掉 -> poster 整行看起来"全白"(分数是在原生 256
    #   上算的, 未受影响)。POSTER_CELL / POSTER_NMAX 允许出可判读的大图。
    _cap = int(os.environ.get("POSTER_NMAX", "0"))
    if _cap > 0:
        n_max = min(n_max, _cap)
    cell = int(os.environ.get("POSTER_CELL", "0")) or max(
        64, min(224, 1280 // max(n_max, 1)))
    input_dir = os.path.join(results_dir, "eval_samples_ctrl", f"{set_name}_input_g")
    has_input = os.path.isdir(input_dir) and bool(glob.glob(os.path.join(input_dir, "g*.png")))

    font = _load_font(r"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", max(14, cell // 8))
    font_small = _font_cache.get(cell) or _load_font(
        r"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", max(9, cell // 7))
    _font_cache[cell] = font_small
    font_big = _load_font(r"/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", max(28, cell // 3))
    label_h = max(40, cell // 4)
    hdr_h = 56

    def _cell(path, bg, placeholder_label="N/A"):
        if path and os.path.exists(path):
            return _Img.open(path).convert("RGB").resize((cell, cell), _Img.LANCZOS)
        img = _Img.new("RGB", (cell, cell), bg)
        d = ImageDraw.Draw(img)
        d.rectangle([1, 1, cell - 2, cell - 2], outline=(55, 60, 70), width=1)
        d.line([(4, 4), (cell - 5, cell - 5)], fill=(45, 48, 55), width=1)
        d.line([(4, cell - 5), (cell - 5, 4)], fill=(45, 48, 55), width=1)
        lbl = placeholder_label or "N/A"
        try:
            bbox = font_small.getbbox(lbl)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            d.text(((cell - tw) // 2, (cell - th) // 2), lbl, font=font_small, fill=(140, 145, 155))
        except Exception:
            pass
        return img

    # ── 主图: input 行 + 每 step gen 行 + GT 行 ──────────────────────────
    _ref = _train_ref_rows(train_csv, eval_csv, n_eval) if train_csv else None
    # ★ 同字最近邻: 用**最新 step** 的生成图
    _nn = (_train_nn_row(train_csv, eval_csv, n_eval, steps[-1][1])
           if (train_csv and steps) else None)
    W = cell * n_max + gap * 2
    n_rows = (len(steps) + 1 + (1 if has_input else 0) + (2 if _ref else 0)
              + (1 if _nn else 0))
    H = hdr_h + gap + n_rows * (label_h + cell + gap) + gap + 30
    canvas = _Img.new("RGB", (W, H), (15, 17, 22))
    draw = ImageDraw.Draw(canvas)
    y = gap
    draw.rectangle([0, y, W, y + hdr_h], fill=(0, 0, 0))
    # ★ 2026-10-05: 无 skel/glyph 条件时 has_input=False, 表头不能写 "input g"
    _in_lbl = "input g" if has_input else "no skel/glyph cond"
    _t = f"{set_name} (n={n_max}) — {_in_lbl} / per-ckpt gen / GT"
    if _ref:
        _t += " / 训练集类似条件 GT"
    if _nn:
        _t += " / 同字最像"
    draw.text((gap, y + 12), _t, font=font, fill=(255, 200, 120))
    y += hdr_h + gap
    if has_input:
        draw.text((gap, y + 8), "input (标准字 g)", font=font_big, fill=(120, 220, 255))
        y += label_h
        for i in range(n_max):
            canvas.paste(_cell(os.path.join(input_dir, f"g{i}.png"), (30, 40, 60)),
                         (gap + i * cell, y))
        y += cell + gap
    for step, d, n in steps:
        draw.text((gap, y + 8), f"step {step}  {_step_ssim_txt(results_dir, step, set_name)}",
                  font=font_big, fill=(160, 200, 255))
        y += label_h
        for i in range(n):
            canvas.paste(_cell(os.path.join(d, f"g{i}.png"), (40, 40, 40)), (gap + i * cell, y))
        y += cell + gap
    draw.text((gap, y + 8), "GT", font=font_big, fill=(255, 160, 160))
    y += label_h
    gt_dir = steps[-1][1]
    for i in range(n_max):
        canvas.paste(_cell(os.path.join(gt_dir, f"gt{i}.png"), (50, 50, 50)), (gap + i * cell, y))
    y += cell + gap
    # ★ 训练集「类似条件」参照两行 —— 用来区分"不会这个字的结构"还是"不会这个书家的笔法"
    if _ref:
        for _lbl, _paths in zip(("训练集·同书家异字", "训练集·同字异书家"), _ref):
            draw.text((gap, y + 8), _lbl, font=font_big, fill=(170, 255, 170))
            y += label_h
            for i in range(n_max):
                _pth = _paths[i] if i < len(_paths) else None
                canvas.paste(_cell(_pth, (25, 25, 25)), (gap + i * cell, y))
            y += cell + gap
    # ★ 训练集「同字最像」一行: 格子角上标出 SSIM
    if _nn:
        _pnn, _snn, _mnn = _nn
        _v = [s for s in _snn if s == s]
        _mean = (sum(_v) / len(_v)) if _v else float("nan")
        draw.text((gap, y + 8),
                  f"训练集·同字最像 (SSIM 均值={_mean:.4f}, n={len(_v)})",
                  font=font_big, fill=(255, 220, 120))
        y += label_h
        _tag_h = max(12, cell // 5)
        for i in range(n_max):
            _pth = _pnn[i] if i < len(_pnn) else None
            canvas.paste(_cell(_pth, (25, 25, 25)), (gap + i * cell, y))
            _s = _snn[i] if i < len(_snn) else float("nan")
            if _s == _s:
                _txt = f"{_s:.2f}"
                draw.rectangle([gap + i * cell, y, gap + i * cell + 6 * len(_txt), y + _tag_h],
                               fill=(0, 0, 0))
                draw.text((gap + i * cell + 2, y + 1), _txt, font=font_small,
                          fill=(255, 220, 120))
        y += cell + gap
        # 数值单独落 CSV（poster 上只是角标, 不方便抄）
        try:
            _cp = os.path.join(results_dir, "posters",
                               f"{set_name}_train_samechar_nn.csv")
            os.makedirs(os.path.dirname(_cp), exist_ok=True)
            _ev_rows = []
            if eval_csv and os.path.exists(eval_csv):
                with open(eval_csv, encoding="utf-8") as _f:
                    _ev_rows = list(csv.DictReader(_f))[:n_eval]
            with open(_cp, "w", newline="", encoding="utf-8") as _f:
                _w = csv.writer(_f)
                _w.writerow(["idx", "eval_char", "eval_calligrapher",
                             "nn_ssim", "nn_char", "nn_calligrapher", "nn_image_path"])
                for i in range(n_max):
                    _r = _ev_rows[i] if i < len(_ev_rows) else {}
                    _m = _mnn[i] if i < len(_mnn) else None
                    _w.writerow([i, _r.get("character", ""), _r.get("calligrapher", ""),
                                 (f"{_snn[i]:.5f}" if i < len(_snn) and _snn[i] == _snn[i] else ""),
                                 (_m[0] if _m else ""), (_m[1] if _m else ""),
                                 (_pnn[i] or "") if i < len(_pnn) else ""])
            print(f"[poster] 同字最像 SSIM 已落盘: {_cp}")
        except Exception as _ce:
            print(f"[poster] 同字最像 CSV 落盘失败: {_ce!r}")

    out = out or os.path.join(results_dir, "posters", f"{set_name}_poster.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    canvas.save(out)

    # ── 结构图: input 行 + 每 step 行 (gen | 模型预测的 aux) + GT 行 ────────
    # 展示**模型自己预测的**结构通道 (aux_*{i}.png, 由 run_in_mem_eval 落盘),
    # 而不是从 gen 图重算 —— 后者只是"生成图的边缘", 无法判断模型是否学对了 aux 目标。
    _aux_cols = []
    for f in sorted(glob.glob(os.path.join(steps[-1][1], "aux_*0.png"))):
        _b = os.path.basename(f)
        _aux_cols.append(_b[len("aux_"):-len("0.png")])
    ncol = 1 + (len(_aux_cols) if _aux_cols else 2)
    W2 = cell * n_max * ncol + gap * 2
    n_rows2 = len(steps) + (1 if has_input else 0) + 1
    H2 = hdr_h + gap + n_rows2 * (label_h + cell + gap) + gap + 30
    canvas2 = _Img.new("RGB", (W2, H2), (15, 17, 22))
    draw2 = ImageDraw.Draw(canvas2)
    y = gap
    draw2.rectangle([0, y, W2, y + hdr_h], fill=(0, 0, 0))
    draw2.text((gap, y + 12),
               f"{set_name} structure — "
               + ("input g / " if has_input else "no skel/glyph cond / ")
               + "per-ckpt gen|"
               + ("|".join(_aux_cols) if _aux_cols
                  else "canny|skel(recomputed from gen, 非输入)")
               + " / GT",
               font=font, fill=(255, 200, 120))
    y += hdr_h + gap
    if has_input:
        draw2.text((gap, y + 8), "input (标准字 g)", font=font_big, fill=(120, 220, 255))
        y += label_h
        for i in range(n_max):
            canvas2.paste(_cell(os.path.join(input_dir, f"g{i}.png"), (30, 40, 60)),
                          (gap + i * ncol * cell, y))
        y += cell + gap
    for step, d, n in steps:
        draw2.text((gap, y + 8), f"step {step}  {_step_ssim_txt(results_dir, step, set_name)}",
                   font=font_big, fill=(160, 200, 255))
        y += label_h
        for i in range(n):
            x = gap + i * ncol * cell
            gen = _cell(os.path.join(d, f"g{i}.png"), (40, 40, 40))
            canvas2.paste(gen, (x, y))
            if _aux_cols:
                for k, an in enumerate(_aux_cols):
                    canvas2.paste(_cell(os.path.join(d, f"aux_{an}{i}.png"), (30, 30, 30)),
                                  (x + (k + 1) * cell, y))
            else:   # 兼容: 老 ckpt 没有 aux 落盘时退回"从 gen 现算"
                canvas2.paste(_poster_canny(gen), (x + cell, y))
                canvas2.paste(_poster_skeleton(gen), (x + 2 * cell, y))
        y += cell + gap
    # GT 行: gt | 目标结构参照 (skel -> 骨架, 其余 -> 边缘; 均从 GT 图现算, 白底黑线)
    draw2.text((gap, y + 8), "GT", font=font_big, fill=(255, 160, 160))
    y += label_h
    _gt_dir2 = steps[-1][1]
    for i in range(n_max):
        x = gap + i * ncol * cell
        gti = _cell(os.path.join(_gt_dir2, f"gt{i}.png"), (50, 50, 50))
        canvas2.paste(gti, (x, y))
        if _aux_cols:
            for k, an in enumerate(_aux_cols):
                ref = _poster_skeleton(gti) if "skel" in an else _poster_canny(gti)
                canvas2.paste(ref, (x + (k + 1) * cell, y))
        else:
            canvas2.paste(_poster_canny(gti), (x + cell, y))
            canvas2.paste(_poster_skeleton(gti), (x + 2 * cell, y))
    out2 = os.path.join(results_dir, "posters", f"{set_name}_struct.png")
    canvas2.save(out2)
    return out
