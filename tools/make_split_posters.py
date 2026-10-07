# -*- coding: utf-8 -*-
"""make_split_posters.py — 三张**客观**海报: Best / Mid / Worst (按 v68 三等分)。

与 tools/render_three_tier_posters.py 的关键区别 (那套不客观的地方):
  1. 指标**全部从 assets/eval200_splits_exact_metrics.csv 读**, 不再在代码里手写常量;
  2. 档位定义改成**固定三份 CSV** (exp-std/csv/eval200_split_{top,mid,worst}.csv,
     由 v68 每样本 SSIM 降序切 62/63/62), 不再用旧的 30 张集;
  3. 图片来源 = 根目录 eval/<model>/{idx}.png (与指标脚本同源);
  4. 不放"现象归因/终极集大成"这类主观话术, 只写可核查的事实 (配置/步数/口径/选择规则);
  5. 每格图片下方标**该样本的 SSIM**(逐图), 行卡片上标**本档内均值** SSIM/LPIPS/IoU。

口径: SSIM = RGB 逐通道高斯窗(win11,σ1.5); LPIPS = alex; IoU = 墨迹 IoU。
      三个数都由 tools/eval_split_metrics.py 用仓库 src/eval 的模块算出。

用法: python tools/make_split_posters.py
输出: docs/04_experiments/imgs/eval200_{best,mid,worst}_split_poster.png
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

OUT_DIR = "docs/04_experiments/imgs"
EVAL_DIR = "eval"
METRICS_CSV = "assets/eval200_splits_exact_metrics.csv"
SPLIT_CSVS = {"best": "exp-std/csv/eval200_split_top.csv",
              "mid": "exp-std/csv/eval200_split_mid.csv",
              "worst": "exp-std/csv/eval200_split_worst.csv",
              # 第 4 档 (用户 2026-10-07): mid ∩ 组合未见 = 38 张
              "midstrict": "exp-std/csv/eval200_split_midstrict.csv"}
PER_SAMPLE = "assets/eval200fix_models_per_sample.csv"
# 每档列数: mid 是主海报 -> 加大到 16; mid-strict 只有 38 张 -> 也 16; best/worst 保持 10
N_COLS_BY_TIER = {"best": 10, "mid": 16, "worst": 10, "midstrict": 16}
N_COLS = 10          # 兼容旧引用 (默认值)

# 行: (目录名, 显示名, 事实描述)
ROWS = [
    ("v68", "v68", "Sp/2 59.2M · 200k步 · batch288 · 3x对称粗细增强 + C2OT"),
    ("moyi_4ch", "Moyi 4ch", "Moyun-4channel-B 252M · 官方实现 (我们复现推理)"),
    ("v66", "v66", "S/2 33.9M · 150k步 · batch384 · 三表 + 条件分层路由(2,4,5,6)"),
    ("v54", "v54", "S/2 33.9M · 150k步 · 三表 minimal (无骨架/无增强)"),
    ("moyi_12ch", "Moyi 12ch", "Moyun-12channel-B 252M · 官方实现 (我们复现推理)"),
    ("v70", "v70", "Sp/2 59.2M · step 30k (训练中) · 增强 + 标准骨架 + C2OT"),
    # ── 48 机器容量阶梯: 已用**我们主线的 Heun 协议 (50步/cfg1.0/seed0)** 在 48 上重导 187 张 ──
    # ⚠ 训练配方与我们不同 (layer/gelu/无RoPE/无QK-norm/无REPA/无C2OT), 已写入行说明与脚注
    ("v_b_aug_route_60k", "B/2 +aug+route",
     "B/2 130M · 60k步 · bs384 · 48配方(无REPA/C2OT) · route 2,4,5,6"),
    ("v_b_aug_40k", "B/2 +aug",
     "B/2 130M · 40k步 · bs384 · 48配方 · 无路由"),
    ("v_l_aug_route_50k", "L/2 +aug+route",
     "L/2 457M · 50k步 · bs128 · 48配方 · route 4,8,10,12"),
    ("v_sp_20k", "Sp/2 (原始数据)",
     "Sp/2 59M · 20k步 · bs580 · 48配方 · 无增强"),
    ("v_b_base_5k", "B/2 (5k 半训)",
     "B/2 130M · 仅 5k步(半训) · bs384 · 48配方 · 无增强"),
]

# 人工视觉排序 (用户 2026-10-07 裁定: 按视觉打分排, 不按指标排)
# 判据 (用户二次裁定): **以 Mid 档为主, Best 档为辅, 不看 Worst 档** —— 即看"日常中间难度"的真实水准。
# 依据: _sync_work/judge_mid_bl.png / judge_mid_top.png / judge_mid_rest.png / judge_best_*.png
#       (每格原始 256px 未缩放)
VISUAL_ORDER = [
    "v_b_aug_route_60k",   # Mid 最贴 GT: 墨量/粗单笔(已)/brushy(类) 都对, 仅 类 略糊
    "v_l_aug_route_50k",   # 与 B60k 同级, 略软
    "v_b_aug_40k",         # 同族, Mid 上 类 更碎一些
    "v68",                 # 干净、飞白好, 但系统性偏淡: 已 太细、类 碎成点
    "v54",                 # Mid 干净贴 GT (略淡); Best 档 眾 发虚
    "v66",                 # 多数不错, 但 Best 档 仙 丢偏旁(结构性失败) + 偶发涂块
    "moyi_4ch",            # 结构基本正确, 但系统性偏淡
    "moyi_12ch",           # 墨量足, 但 Mid 上 訟 结构碎裂
    "v_sp_20k",            # 涂抹/断裂
    "v70",                 # 30k 半程, 结构性碎裂
    "v_b_base_5k",         # 5k 半训, 最差
]

CELL, PAD = 190, 12
ROW_HDR_W, COL_HDR_H, BANNER_H, FOOTER_H = 350, 150, 150, 270


def font(sz, bold=False):
    for p in ("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
              "C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


F = {"title": font(30, True), "sub": font(15), "pill": font(13),
     "row_t": font(18, True), "row_s": font(12), "mk": font(10), "mv": font(16, True),
     "col_ch": font(34, True), "col": font(13, True), "col_s": font(11),
     "cell": font(11), "foot": font(12)}


def load_splits():
    """读固定三份 CSV -> {tier: [idx...]}, 并用每样本 CSV 重建结果交叉核对。"""
    ids = {}
    with open("exp-std/csv/eval200_fixed.csv", encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f)):
            ids[r["image_path"]] = i

    out = {}
    for tier, p in SPLIT_CSVS.items():
        with open(p, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        idxs = [ids[r["image_path"]] for r in rows]
        out[tier] = idxs

    # 交叉核对: 用 v68 每样本 ssim 重建, 应与固定 CSV 完全一致
    per = list(csv.DictReader(open(PER_SAMPLE, encoding="utf-8")))
    v68 = sorted([(float(r["ssim"]), int(r["idx"])) for r in per if r["model_name"] == "v68"],
                 key=lambda x: -x[0])
    rebuild = {"best": [i for _, i in v68[0:62]],
               "mid": [i for _, i in v68[62:125]],
               "worst": [i for _, i in v68[125:187]]}
    for tier in ("best", "mid", "worst"):
        same = set(out[tier]) == set(rebuild[tier])
        print(f"  [split] {tier:9s} n={len(out[tier]):3d}  与 per-sample 重建一致: "
              f"{'✓' if same else '✗ 不一致!'}")
    # mid-strict 不来自 per-sample 重建, 而是 mid ∩ 组合未见 —— 直接核验其定义
    K3 = ("calligrapher", "script", "character")
    seen3 = {tuple(str(r.get(k, "")).strip() for k in K3)
             for r in csv.DictReader(open("exp-std/csv/train.csv", encoding="utf-8"))}
    evi = list(csv.DictReader(open("exp-std/csv/eval200_fixed.csv", encoding="utf-8")))
    un_idx = {i for i, r in enumerate(evi)
              if tuple(str(r.get(k, "")).strip() for k in K3) not in seen3}
    ms_ok = set(out["midstrict"]) == (set(out["mid"]) & un_idx)
    any_seen = any(tuple(str(evi[i].get(k, "")).strip() for k in K3) in seen3
                   for i in out["midstrict"])
    print(f"  [split] midstrict n={len(out['midstrict']):3d}  = mid ∩ 组合未见: "
          f"{'✓' if ms_ok else '✗'}   档内含组合见过的样本: {any_seen} (应为 False)")
    return out, {int(r["idx"]): r for r in per if r["model_name"] == "v68"}


def load_metrics():
    return {r["model"]: r for r in csv.DictReader(open(METRICS_CSV, encoding="utf-8"))}


def load_leakage():
    """返回 (idx->组合是否在训练里出现过, {model: {idx: 逐样本SSIM}})。

    样本级无泄漏 (img_id/image_path/src_image_path 与 train.csv 0 重叠, 见
    tools/audit_eval_leakage.py), 但 (书家,书体,字) 组合有 75/187 命中 —— 必须披露。
    """
    K3 = ("calligrapher", "script", "character")
    seen = {tuple(str(r.get(k, "")).strip() for k in K3)
            for r in csv.DictReader(open("exp-std/csv/train.csv", encoding="utf-8"))}
    ev = list(csv.DictReader(open("exp-std/csv/eval200_fixed.csv", encoding="utf-8")))
    idx_seen = {i: tuple(str(r.get(k, "")).strip() for k in K3) in seen
                for i, r in enumerate(ev)}
    per = list(csv.DictReader(open(PER_SAMPLE, encoding="utf-8")))
    by_model = {}
    for r in per:
        by_model.setdefault(r["model_name"], {})[int(r["idx"])] = float(r["ssim"])
    return idx_seen, by_model


def combo_stats(tier_idxs, idx_seen, by_model):
    """本档内: 组合见过占比 + v68/v66 在 见过/未见 两子集上的 SSIM 差。"""
    a = [i for i in tier_idxs if idx_seen[i]]
    b = [i for i in tier_idxs if not idx_seen[i]]
    out = {"n_seen": len(a), "n": len(tier_idxs),
           "share": len(a) / max(1, len(tier_idxs)) * 100}
    for m in ("v68", "v66"):
        d = by_model.get(m, {})
        if a and b and all(i in d for i in a + b):
            ma = sum(d[i] for i in a) / len(a)
            mb = sum(d[i] for i in b) / len(b)
            out[m] = (ma, mb, ma - mb)
    return out


def pick_cols(tier_idxs, v68meta, ncols=None):
    """本档内按 v68 SSIM 降序等距取 ncols 个 (确定性, 非挑图)。"""
    ncols = int(ncols or N_COLS)
    ranked = sorted(tier_idxs, key=lambda i: -float(v68meta[i]["ssim"]))
    n = len(ranked)
    if n <= ncols:
        return ranked
    pos = [round(k * (n - 1) / (ncols - 1)) for k in range(ncols)]
    return [ranked[p] for p in pos]


def ssim_1(idx, model):
    """逐图 SSIM (与行内同口径), 只算海报上真正显示的那 10 格。"""
    from src.eval.metrics import ssim as our_ssim
    p = os.path.join(EVAL_DIR, model, f"{idx}.png")
    g = os.path.join(EVAL_DIR, "gt", f"{idx}.png")
    if not (os.path.exists(p) and os.path.exists(g)):
        return None
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    b = np.asarray(Image.open(g).convert("RGB"), dtype=np.float32) / 255.0
    return float(our_ssim(a, b))


TIER_META = {
    "best":  ("Best",  62, "#10b981", "本档 = v68 每样本 SSIM 降序的第 1–62 名"),
    "mid":   ("Mid",   63, "#38bdf8", "本档 = v68 每样本 SSIM 降序的第 63–125 名"),
    "worst": ("Worst", 62, "#f43f5e", "本档 = v68 每样本 SSIM 降序的第 126–187 名"),
    "midstrict": ("Mid-Strict", 38, "#a78bfa",
                  "本档 = mid 档 ∩ **组合未见** (剔掉 (书家,书体,字) 在训练里出现过的 25 张)"),
}


def render(tier, tier_idxs, v68meta, metrics, idx_seen, by_model):
    name, n, theme, split_desc = TIER_META[tier]
    # 指标 CSV 的列名前缀 (海报上的 best 对应 CSV 里的 top)
    mkey = {"best": "top", "mid": "mid", "worst": "worst", "midstrict": "midstrict"}[tier]
    ncols = N_COLS_BY_TIER.get(tier, N_COLS)
    cols = pick_cols(tier_idxs, v68meta, ncols)
    # 行序 = **人工视觉排序** (2026-10-07 用户裁定), 判据见脚注⑤。
    # 未列入 VISUAL_ORDER 的行按全 187 SSIM 降序补在后面, 不会丢。
    idx_of = {r[0]: i for i, r in enumerate(ROWS)}
    listed = [idx_of[m] for m in VISUAL_ORDER if m in idx_of]
    rest = [i for i in range(len(ROWS)) if i not in listed]
    rest.sort(key=lambda i: -(float(metrics[ROWS[i][0]]["all_ssim"])
                              if ROWS[i][0] in metrics else 0.0))
    order = [ROWS[i] for i in listed + rest]
    rows = [("gt", "真迹 GT", "eval200_fixed 原始真迹切片 (参照行, 不参与指标)")] + order

    canvas_w = ROW_HDR_W + ncols * (CELL + PAD) + PAD * 2
    canvas_h = BANNER_H + COL_HDR_H + len(rows) * (CELL + PAD) + FOOTER_H
    img = Image.new("RGB", (canvas_w, canvas_h), "#090c10")
    d = ImageDraw.Draw(img)

    # ---- Banner ----
    d.rectangle([(0, 0), (canvas_w, BANNER_H)], fill="#0e131d")
    d.line([(PAD * 2, BANNER_H - 2), (canvas_w - PAD * 2, BANNER_H - 2)], fill=theme, width=2)
    d.text((PAD * 2, 20), f"eval200_fixed · {name} 档 (n={n}) — 同字横向对照", font=F["title"], fill="#f8fafc")
    d.text((PAD * 2, 64),
           f"{split_desc}   ｜   列=本档内按 v68 SSIM 降序等距取 {ncols} 个 (确定性, 非挑选)   "
           f"｜   行内 SSIM/LPIPS/IoU = **本档内均值**; 每格下方 = 该样本的 SSIM",
           font=F["sub"], fill="#94a3b8")
    pills = [("口径: SSIM(RGB 高斯窗) / LPIPS(alex) / IoU(墨迹)", "#334155", "#1e293b"),
             (f"行内均值样本数 n={n}", "#334155", "#1e293b"),
             ("来源: tools/eval_split_metrics.py (复用 src/eval 模块)", "#334155", "#1e293b")]
    if tier == "midstrict":
        pills.insert(0, ("★ 严格档: 剔除 (书家,书体,字) 组合见过者 (combine-seen 0/38)",
                         "#7c3aed", "#3b0764"))
    x = PAD * 2
    for txt, bd, bg in pills:
        w = int(d.textlength(txt, font=F["pill"])) + 22
        d.rounded_rectangle([(x, 100), (x + w, 130)], radius=6, fill=bg, outline=bd)
        d.text((x + 11, 106), txt, font=F["pill"], fill="#e2e8f0")
        x += w + 12

    # ---- 列头 (字/书家/书体/ID/v68 SSIM) ----
    for ci, idx in enumerate(cols):
        cx = ROW_HDR_W + PAD + ci * (CELL + PAD)
        cy = BANNER_H + 10
        m = v68meta[idx]
        d.rounded_rectangle([(cx, cy), (cx + CELL, cy + COL_HDR_H - 18)],
                            radius=8, fill="#121824", outline="#1e293b")
        d.text((cx + 12, cy + 8), m["char"], font=F["col_ch"], fill="#facc15")
        d.text((cx + 74, cy + 12), f"{m['calligrapher']} · {m['script']}", font=F["col"], fill="#f1f5f9")
        d.text((cx + 12, cy + 56), f"ID #{idx:03d}", font=F["col_s"], fill="#64748b")
        d.text((cx + 12, cy + 76), f"v68 SSIM {float(m['ssim']):.4f}", font=F["col_s"], fill=theme)
        d.text((cx + 12, cy + 98), "本档内排名见下", font=F["col_s"], fill="#475569")

    # ---- 行 ----
    y0 = BANNER_H + COL_HDR_H
    for ri, (mid, disp, desc) in enumerate(rows):
        y = y0 + ri * (CELL + PAD)
        is_gt = mid == "gt"
        d.rounded_rectangle([(PAD, y), (ROW_HDR_W - PAD, y + CELL)], radius=8,
                            fill="#1e1b18" if is_gt else "#121824",
                            outline="#eab308" if is_gt else "#1e293b",
                            width=2 if is_gt else 1)
        d.text((PAD + 16, y + 12), disp, font=F["row_t"], fill="#fef08a" if is_gt else "#f8fafc")
        # 描述可能较长, 手动折行
        words, line, lines = desc.split(" "), "", []
        for w_ in words:
            if d.textlength(line + " " + w_, font=F["row_s"]) > ROW_HDR_W - 2 * PAD - 32:
                lines.append(line); line = w_
            else:
                line = (line + " " + w_).strip()
        lines.append(line)
        for k, ln in enumerate(lines[:2]):
            d.text((PAD + 16, y + 38 + k * 17), ln, font=F["row_s"], fill="#94a3b8")

        # 三个指标盒 (本档内均值)
        mx, my = PAD + 16, y + 86
        if is_gt:
            d.rounded_rectangle([(mx, my), (mx + 300, my + 50)], radius=5,
                                fill="#0b0f17", outline="#1e293b")
            d.text((mx + 10, my + 8), "参照行", font=F["mk"], fill="#64748b")
            d.text((mx + 10, my + 26), "SSIM / LPIPS / IoU 不适用", font=F["mv"], fill="#94a3b8")
        else:
            r = metrics[mid]
            vals = [("SSIM ↑", f"{float(r[f'{mkey}_ssim']):.4f}", "#34d399"),
                    ("LPIPS ↓", f"{float(r[f'{mkey}_lpips']):.4f}", "#a78bfa"),
                    ("IoU ↑", f"{float(r[f'{mkey}_iou']):.4f}", "#f59e0b")]
            for k, v, c in vals:
                d.rounded_rectangle([(mx, my), (mx + 96, my + 50)], radius=5,
                                    fill="#0b0f17", outline="#1e293b")
                d.text((mx + 8, my + 7), k, font=F["mk"], fill="#64748b")
                d.text((mx + 8, my + 24), v, font=F["mv"], fill=c)
                mx += 102
            d.text((PAD + 16, y + CELL - 22),
                   f"本档 n={int(r[f'{mkey}_n'])}  ← 均值口径", font=F["cell"], fill="#475569")

        # ---- 格子 ----
        for ci, idx in enumerate(cols):
            cx = ROW_HDR_W + PAD + ci * (CELL + PAD)
            d.rounded_rectangle([(cx, y), (cx + CELL, y + CELL)], radius=6,
                                fill="#0e131d", outline="#1e293b")
            p = os.path.join(EVAL_DIR, ("gt" if is_gt else mid), f"{idx}.png")
            if os.path.exists(p):
                im = Image.open(p).convert("RGB").resize((CELL - 20, CELL - 20), Image.LANCZOS)
                img.paste(im, (cx + 10, y + 7))
                if not is_gt:
                    s = ssim_1(idx, mid)
                    d.rectangle([(cx + 2, y + CELL - 20), (cx + CELL - 2, y + CELL - 2)],
                                fill="#090c10")
                    d.text((cx + 7, y + CELL - 18),
                           f"SSIM {s:.4f}" if s is not None else "缺失", font=F["cell"], fill="#94a3b8")
            else:
                d.text((cx + 40, y + CELL // 2), "无图", font=F["cell"], fill="#475569")

    # ---- Footer ----
    yf = canvas_h - FOOTER_H + 14
    d.line([(PAD * 2, yf - 10), (canvas_w - PAD * 2, yf - 10)], fill="#1e293b")
    notes = [
        "① 三档由**固定文件**定义: exp-std/csv/eval200_split_{top,mid,worst}.csv (v68 每样本 SSIM 降序 62/63/62), 三张海报共用同一套档位划分。",
        "② 行内 SSIM/LPIPS/IoU 均为**该档内的均值**(不是全 187 张), 取自 assets/eval200_splits_exact_metrics.csv。",
        "③ Moyi 两行为**我们按统一口径复现推理**的结果, 与其官方论文/报告里的数字不可直接比 (推理协议不同)。",
        "④ v13 / v21 / v23 只有 38 张样本, 无法参与 62/63/62 分档, 故不列入本海报。",
        "⑤ 行序 = **人工视觉排序** (用户裁定, 非指标排序; 判据: 以 Mid 档为主、Best 档为辅, 不看 Worst), "
        "依据评审图 judge_mid_*.png (每格原始 256px 未缩放)。"
        "⚠ 指标盒里的数字仍是**实测值**, 与行序无关 —— 两者不一致处正是 SSIM 与视觉的分离点。",
        "⑥ ⚠ 档位是**相对 v68 定义**的: 因此 v68 在 Best 档天然占优、在 Worst 档天然吃亏; 跨档比较应在同一档内进行。",
    ]
    st = combo_stats(tier_idxs, idx_seen, by_model)
    notes.append(
        f"⑦ 泄漏审计 (tools/audit_eval_leakage*.py): 样本级**零重叠** —— img_id / image_path / "
        f"src_image_path 与训练清单全不命中; 但 (书家,书体,字) **组合级**: 本档 {st['n_seen']}/{st['n']} "
        f"张 ({st['share']:.0f}%) 的组合在训练里出现过 (全 187 张是 75/187 = 40%)。")
    if "v68" in st and "v66" in st:
        notes.append(
            f"⑧ 组合「见过」子集对**所有模型**都更容易: 本档内 v68 {st['v68'][2]:+.4f} / "
            f"v66 {st['v66'][2]:+.4f}; 全 187 张 v68 +0.0576 / v66 +0.0771 (v66 偏差更大 ⇒ 是难度效应, "
            f"不是 v68 背答案)。故跨模型比较仍成立, 但**绝对值偏乐观**, 不可读作「严格未见字」。")
        notes.append(
            "⑨ 严格口径备选: 组合未见的 112 张见 exp-std/csv/eval200_combo_unseen.csv "
            "(项目自带的 eval_top10_strict_subset84.csv 并不更严格: 组合见过 75/84 = 89%)。")
    notes.append(
        "⑩ 48 阶梯各行 (B/2, L/2, Sp/2): **生成协议与我们相同** (50步Heun / cfg1.0 / seed0, "
        "在 48 上由 tools/dump_eval187_for48.py 重导 187 张), 故可与其他行同口径比较; "
        "⚠ 但**训练配方不同** (layer/gelu/无RoPE/无QK-norm/无REPA/无C2OT, lr 1.5e-4|1e-4) —— "
        "它们与 v54/v66/v68 是「尺寸+配方」共同改变, 不是纯容量对照。")
    notes.append(
        "⑪ 48 自己报告的数字用的是 **50步 Euler + LPIPS(vgg)**, 与本海报的 Heun + LPIPS(alex) 不同口径, "
        "不可直接对照 (例: B/2@60k 它报 0.6104, 本海报口径 0.5964)。")
    for k, t in enumerate(notes):
        d.text((PAD * 2, yf + k * 20), t, font=F["foot"], fill="#94a3b8")

    out = os.path.join(OUT_DIR, f"eval200_{tier}_split_poster.png")
    img.save(out)
    print(f"✓ {out}  ({canvas_w}×{canvas_h})")
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("[load] 固定四档 + 交叉核对")
    splits, v68meta = load_splits()
    metrics = load_metrics()
    idx_seen, by_model = load_leakage()
    print(f"[load] 指标行数 = {len(metrics)}: {sorted(metrics)}")
    print(f"[load] 组合级「见过」: 全 187 张里 {sum(idx_seen.values())} 张")
    # 渲染顺序: **mid 在前 (主海报)**, 然后 mid-strict, 再 best/worst
    for tier in ("mid", "midstrict", "best", "worst"):
        st = combo_stats(splits[tier], idx_seen, by_model)
        print(f"[load] {tier:10s} n={st['n']:3d} 组合见过 {st['n_seen']:3d} "
              f"({st['share']:.0f}%)" +
              (f"  v68 Δ={st['v68'][2]:+.4f} v66 Δ={st['v66'][2]:+.4f}" if "v68" in st else ""))
        render(tier, splits[tier], v68meta, metrics, idx_seen, by_model)
    print("完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
