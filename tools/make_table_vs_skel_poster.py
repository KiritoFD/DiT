# -*- coding: utf-8 -*-
"""make_table_vs_skel_poster.py — 「汉字表 vs std skel」同频对照海报。

素材: _sync_work/poster_src/{gt,v68,v70}/<idx>.png  (由 4090 上
      eval_samples_ctrl/step0075000|step0050000/eval200fix 里 {g,gt}<idx>.png 抓来)
数据: exp-std/csv/eval200_fixed.csv (列 idx -> 书家/书体/字)

口径: 每格像素来自模型 eval 落盘图 (8-bit PNG), 逐格 SSIM 由本地 src.eval.metrics.ssim 算,
      因此"本 12 格均值"与训练期 in-mem eval 的全 187 均值会有小幅出入(8bit 量化 + 子集),
      两者都写在图上, 不混淆。

用法: python tools/make_table_vs_skel_poster.py
输出: docs/107/imgs/table_vs_skel_75000.png
"""
import csv
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.eval.metrics import ssim as our_ssim  # noqa: E402

SRC = "_sync_work/poster_src"
OUT_DIR = "docs/107/imgs"
EVAL_CSV = "exp-std/csv/eval200_fixed.csv"

# 训练期 in-mem eval 的真实全 187 数字（本次报告核实过）
INMEM = {
    ("v68", "5k"):  (0.5220, 0.4900),
    ("v70", "5k"):  (0.5341, 0.5043),
    ("v68", "75k"): (0.5889, 0.6206),
    ("v70", "75k"): (0.5520, 0.5613),
}

ROWS = [("gt", "真迹 GT", "g"), ("v68", "v68 字表 @5k", "k5"),
        ("v70", "v70 std-skel @5k", "k5"), ("v68", "v68 字表 @75k", ""),
        ("v70", "v70 std-skel @75k", "")]

CELL, PAD = 168, 8
HDR_W, TOP_H, ROWH, FOOT_H = 250, 150, 118, 320


def font(sz, bold=False):
    for p in ("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
              "C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


F = {"title": font(30, True), "sub": font(15), "row": font(15, True), "row_s": font(12),
     "cell": font(12), "char": font(24, True), "col": font(12), "foot": font(13),
     "num": font(14, True)}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    meta = list(csv.DictReader(open(EVAL_CSV, encoding="utf-8")))
    ks = sorted(int(f.split(".")[0]) for f in os.listdir(f"{SRC}/gt"))

    def L(p):
        with Image.open(p) as im:
            a = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        return a

    W = HDR_W + len(ks) * (CELL + PAD) + PAD * 2
    H = TOP_H + ROWH + 4 * (CELL + PAD + 34) + FOOT_H
    img = Image.new("RGB", (W, H), "#090c10")
    d = ImageDraw.Draw(img)

    # ---- 标题 ----
    d.rectangle([(0, 0), (W, TOP_H)], fill="#0e131d")
    d.line([(PAD * 2, TOP_H - 2), (W - PAD * 2, TOP_H - 2)], fill="#f59e0b", width=2)
    d.text((PAD * 2, 18), "eval200_fixed · 「汉字表」 vs 「std skel」 同频对照",
           font=F["title"], fill="#f8fafc")
    d.text((PAD * 2, 60),
           "v68 与 v70 配置逐项相同: DiT-2Cond-Sp/2 59.14M · 3x对称增强 · C2OT(slot) · "
           "cond_inject_at 2,4,5,6 · rms+swiglu+qk1+rope1 · bs288 · 200k · lr5e-5 cosine · REPA0.03",
           font=F["sub"], fill="#94a3b8")
    d.text((PAD * 2, 86),
           "唯一变量: v68 = 汉字表 4690x256 (no_char_cond=false)   |   "
           "v70 = std skel latent (no_char_cond=true, skel_as_glyph_cond, glyph_inject 2,4,5,6, scale_init 0.6)",
           font=F["sub"], fill="#f59e0b")
    d.text((PAD * 2, 112),
           "列 = 187 张里**等距取 12 张**(索引 0,16,33,…,181; 确定性, 非挑图) · "
           "每格下方 = 该样本的 SSIM(本地 src.eval.metrics.ssim 算, 8-bit PNG)",
           font=F["sub"], fill="#64748b")

    # ---- 列头 (字 + 书家/书体) ----
    y0 = TOP_H
    for j, k in enumerate(ks):
        x = HDR_W + PAD + j * (CELL + PAD)
        ch = meta[k]["character"]
        d.text((x + CELL // 2 - 12, y0 + 4), ch, font=F["char"], fill="#e2e8f0")
        d.text((x + 6, y0 + 40), f"#{k}", font=F["col"], fill="#64748b")
        d.text((x + 6, y0 + 56), f"{meta[k]['calligrapher']}·{meta[k]['script']}",
               font=F["col"], fill="#475569")

    # ---- 行 ----
    y = y0 + ROWH
    for tag, label, pref in ROWS:
        row = []
        for k in ks:
            name = f"{pref}{k}.png" if pref else f"{k}.png"
            p = os.path.join(SRC, tag, name)
            if not os.path.exists(p):
                p = os.path.join(SRC, tag, f"{k}.png")
            row.append(p if os.path.exists(p) else None)
        d.text((PAD * 2, y + 6), label, font=F["row"], fill="#f8fafc" if tag != "gt" else "#a3e635")
        # 全 187 训练期数字
        if tag != "gt":
            time_tag = "5k" if pref == "k5" else "75k"
            if (tag, time_tag) in INMEM:
                ev, sn = INMEM[(tag, time_tag)]
                d.text((PAD * 2, y + 28), f"全187 eval200fix SSIM {ev:.4f}", font=F["row_s"], fill="#94a3b8")
                d.text((PAD * 2, y + 46), f"全187 seen      SSIM {sn:.4f}", font=F["row_s"], fill="#64748b")
        ssims = []
        for j, p in enumerate(row):
            x = HDR_W + PAD + j * (CELL + PAD)
            if p is None:
                d.rectangle([(x, y), (x + CELL, y + CELL)], fill="#1e293b")
                continue
            im = Image.open(p).convert("RGB").resize((CELL, CELL), Image.LANCZOS)
            img.paste(im, (x, y))
            d.rectangle([(x, y), (x + CELL - 1, y + CELL - 1)], outline="#334155")
            gp = os.path.join(SRC, "gt", f"{ks[j]}.png")
            if tag != "gt" and os.path.exists(gp):
                s = float(our_ssim(L(p), L(gp)))
                ssims.append(s)
                col = "#4ade80" if s >= 0.6 else ("#fbbf24" if s >= 0.5 else "#f87171")
                d.text((x + 4, y + CELL + 2), f"{s:.3f}", font=F["cell"], fill=col)
        if ssims:
            d.text((PAD * 2, y + 68), f"本12格均值 {np.mean(ssims):.4f}", font=F["row_s"], fill="#fbbf24")
        y += CELL + PAD + 34

    # ---- 页脚: 轨迹与结论 ----
    fy = y + 6
    d.line([(PAD * 2, fy), (W - PAD * 2, fy)], fill="#334155", width=1)
    d.text((PAD * 2, fy + 10), "① 同频轨迹 (in-mem eval, 全 187 / seen 20)", font=F["row"], fill="#f8fafc")
    traj = [
        "step | v70 eval200fix | v68 eval200fix |   Δ    | v70 seen | v68 seen |   Δ",
        "  5k |    0.5341      |    0.5220      | +0.0121 |  0.5043  |  0.4900  | +0.0143",
        " 10k |    0.5349      |    0.5297      | +0.0052 |  0.5031  |  0.5057  | -0.0026",
        " 20k |    0.5429      |    0.5429      |  0.0000 |  0.5132  |  0.5249  | -0.0117",
        " 40k |    0.5466      |    0.5621      | -0.0155 |  0.5252  |  0.5586  | -0.0334",
        " 60k |    0.5535      |    0.5791      | -0.0256 |  0.5320  |  0.5955  | -0.0635",
        " 75k |    0.5520      |    0.5889      | -0.0369 |  0.5613  |  0.6206  | -0.0593",
        "200k |       —        |    0.6149      |    —    |    —     |  0.7021  |    —",
    ]
    yy = fy + 34
    for t in traj:
        d.text((PAD * 2 + 6, yy), t, font=F["foot"], fill="#cbd5e1" if t[0] != "s" else "#94a3b8")
        yy += 19

    xx = W // 2 + 40
    d.text((xx, fy + 10), "② 读出来的三件事", font=F["row"], fill="#f8fafc")
    notes = [
        "1) std skel 冷启动更快: 前 20k 步领先(5k 时 +0.012)，20k 起被反超;",
        "   到 75k 已落后 -0.0369，且斜率没有收窄 -> 天花板明显更低。",
        "2) v70 已接近平台: 50k->75k(25k 步)只涨 +0.0024 -> 终点预计 0.56~0.57，",
        "   v68 同期 0.5889、终点 0.6149。",
        "3) 记忆 vs 泛化: seen-eval200fix 差  v70 = +0.009  vs  v68 = +0.032;",
        "   v68 到 200k 拉到 +0.087。std skel 无字符身份可背 -> 无记忆红利(也是开集来源)。",
        "",
        "⚠ 最需警惕: 碎片率 frag   v70  2.16(10k)->3.38(50k)->5.28(75k) 单调恶化;",
        "   v68 稳定在 2.5 附近。std skel 只给位置/形状，不给笔画连通性 -> 笔画被打散。",
    ]
    for i, t in enumerate(notes):
        d.text((xx, fy + 34 + i * 19), t, font=F["foot"],
               fill="#f87171" if "⚠" in t or "落后" in t else "#cbd5e1")

    out = os.path.join(OUT_DIR, "table_vs_skel_75000.png")
    img.save(out)
    print(f"✓ {out}  ({W}x{H})")


if __name__ == "__main__":
    main()
