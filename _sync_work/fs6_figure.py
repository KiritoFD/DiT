#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""fs6_figure.py — few-shot 的两种图：每主题四联 poster + 跨主题同字对比。

四联（每个主题一张）: 骨架 g | GT 真迹 | 零梯度直写(row_pt) | 训练 2000 步
  -> 回答"训练到底改变了肉眼可见的什么"。
跨主题同字（一张）: 同一个字、同一套骨架，5 个新书家各自的风格行生成出来的样子
  -> 这是"风格行真的在控制风格"最直接的证据（ssim 看不出来，ssim 主要看字对不对）。

用法: /opt/conda/envs/cu121/bin/python _sync_work/fs6_figure.py
"""
import csv
import glob
import os
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
OUT = "_review/posters/fs6figs"
os.makedirs(OUT, exist_ok=True)
TOPICS = ["沈周-行", "伊秉绶-行", "傅山-行", "伊秉绶-隶", "徐渭-行"]
CELL = 150


def font(size=18):
    """找一张能显示中文的字体；找不到就退回默认（标签改用英文，避免豆腐块）。"""
    cands = (glob.glob("/usr/share/fonts/**/NotoSansCJK*", recursive=True)
             + glob.glob("/usr/share/fonts/**/*wqy*", recursive=True)
             + glob.glob("/usr/share/fonts/**/*.tt[fc]", recursive=True))
    for c in cands:
        try:
            f = ImageFont.truetype(c, size)
            if f.getbbox("书")[2] > 0:          # 真含 CJK 字形
                return f, True
        except Exception:
            continue
    return ImageFont.load_default(), False


FONT, CJK = font(18)
LAB = (["骨架 g", "GT 真迹", "零梯度直写", "训练 2000 步"] if CJK
       else ["skeleton g", "GT (real ink)", "row written, 0 grad", "fine-tuned 2000"])
print(f"[font] CJK={CJK}")


def locate(pattern):
    """-> (最后一步的 fewshot 图目录, 该 run 的 eval_stdskel_batch.csv)。

    图在 <run>/eval_samples_ctrl/stepXXXXXX/fewshot/g*.png，
    而逐样本 CSV 在 <run>/ 下面（eval_only/ 或 run 根）—— 两者不在同一层，
    所以分开找，别拿图目录去找 CSV（那样永远找不到，图就会整张空白）。
    """
    imgs = sorted(glob.glob(os.path.join(pattern, "**", "step*", "fewshot", "g*.png"),
                            recursive=True))
    csvs = sorted(glob.glob(os.path.join(pattern, "**", "eval_stdskel_batch.csv"),
                            recursive=True), key=os.path.getmtime)
    if not imgs or not csvs:
        return None, None
    return os.path.dirname(os.path.dirname(imgs[-1])), csvs[-1]


def rows_of(topic, kind="eval"):
    return list(csv.DictReader(open(f"assets/fs6_{topic}_{kind}.csv", encoding="utf-8")))


def char_idx_from_csv(path):
    """逐样本 CSV 里 fewshot 那批 (char -> 行号)，行号即 g<行号>.png。"""
    seen, out = set(), {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r["set"] != "fewshot":
            continue
        i = int(r["idx"])
        if r["char"] not in seen:
            seen.add(r["char"])
            out[r["char"]] = i
    return out


def paste(im):
    if im and os.path.isfile(im):
        im2 = Image.open(im).convert("L").resize((CELL, CELL), Image.LANCZOS).convert("RGB")
        im2.paste(Image.new("RGB", (2, CELL), (255, 0, 0)), (0, 0))   # 红条=生成图标记
        return im2
    return Image.new("RGB", (CELL, CELL), (235, 235, 235))


# ── 1) 每主题四联 ─────────────────────────────────────────────────────────────
def per_topic(topic, n=10):
    base_dir, base_csv = locate(f"/tmp/_fs6_{topic}_base_row_pt")
    tr_dir, tr_csv = locate(f"assets/results/v15_fs6_{topic}_row_pt_lr0.001*")
    if not base_dir or not tr_dir:
        print(f"[skip] {topic}: base={base_dir} trained={tr_dir}")
        return None
    ba_idx, tr_idx = char_idx_from_csv(base_csv), char_idx_from_csv(tr_csv)
    ev = {r["character"]: r for r in rows_of(topic)}
    chars = [c for c in tr_idx if c in ba_idx and c in ev][:n]
    if not chars:
        print(f"[skip] {topic}: 训练/基线没有共同的字 (tr={len(tr_idx)} ba={len(ba_idx)})")
        return None
    sheet = Image.new("RGB", (len(chars) * CELL, 4 * CELL + 26), "white")
    dr = ImageDraw.Draw(sheet)
    for j, ch in enumerate(chars):
        r = ev[ch]
        sheet.paste(paste(r["std_path"]), (j * CELL, 26))
        sheet.paste(paste(r["image_path"]), (j * CELL, CELL + 26))
        sheet.paste(paste(os.path.join(base_dir, "fewshot", f"g{ba_idx[ch]}.png")),
                    (j * CELL, 2 * CELL + 26))
        sheet.paste(paste(os.path.join(tr_dir, "fewshot", f"g{tr_idx[ch]}.png")),
                    (j * CELL, 3 * CELL + 26))
    for y, lab in enumerate(LAB):
        dr.text((4, y * CELL + 6), lab, fill="red", font=FONT)
    dr.text((4, 4 * CELL + 8),
            f"{topic}   n={len(chars)} 字   (左→右同一批字)" if CJK else
            f"{topic}   {len(chars)} held-out chars", fill="black")
    out = f"{OUT}/poster_{topic.replace('-', '_')}.png"
    sheet.save(out)
    print(out)
    return out


# ── 2) 跨主题同字 ─────────────────────────────────────────────────────────────
def cross_topic(n_chars=8):
    """5 个主题的 eval 集是**互不相交**的（各自从各自能借到 std 的字里抽 100 个，
    405 个字里各抽 100 → 五交合理就是 0）。所以退一步：在 3/4 主题的子集里找
    共有字最多的一组，同一字并排看"不同书家的手"。"""
    from itertools import combinations
    sets = {t: {r["character"] for r in rows_of(t)} for t in TOPICS}
    runs = {}
    for t in TOPICS:
        d, c = locate(f"assets/results/v15_fs6_{t}_row_pt_lr0.001*")
        if d and c:
            runs[t] = (d, char_idx_from_csv(c))
    best = None
    for k in (4, 3, 2):
        for combo in combinations(sorted(runs), k):
            common = set.intersection(*[sets[t] for t in combo])
            n_ok = len(common & set(runs[combo[0]][1]))
            if not best or n_ok > best[2]:
                best = (combo, common, n_ok)
    combo, common, n_ok = best
    print(f"[cross] 5 主题全交=0；改用 {len(combo)} 主题子集 {combo}，共有字 {len(common)}")
    if not common:
        return None
    chars = sorted(common)[:n_chars]
    cols = len(chars) + 1
    sheet = Image.new("RGB", (cols * CELL, len(combo) * 2 * CELL + 30), "white")
    dr = ImageDraw.Draw(sheet)
    y = 30
    for t in combo:
        d, idx = runs[t]
        ev = {r["character"]: r for r in rows_of(t)}
        dr.text((4, y + CELL // 2 - 8), t, fill="blue", font=FONT)
        dr.text((CELL + 4, y - 14), "GT", fill="gray")
        dr.text((CELL + 4, y + CELL - 14), "GEN", fill="gray")
        for j, ch in enumerate(chars):
            if ch not in ev or ch not in idx:
                continue
            sheet.paste(paste(ev[ch]["image_path"]), ((j + 1) * CELL, y))
            sheet.paste(paste(os.path.join(d, "fewshot", f"g{idx[ch]}.png")),
                        ((j + 1) * CELL, y + CELL))
        y += 2 * CELL
    out = f"{OUT}/cross_topic_same_char.png"
    sheet.save(out)
    print(out, "字:", "".join(chars))
    return out


if __name__ == "__main__":
    for t in TOPICS:
        per_topic(t)
    cross_topic()
