#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""build_font_slot_dataset.py —— **字体槽**数据集（用户方案 ✓）。

用户想法（原文 ✓）
    "把几个标准字体作为**槽位**，参与对比学习（判别器我建议也用**对比学习**），
     应该能有更好的区分度。"

与仓库已有的 `build_coslot_5944_dataset.py` 的**根本区别** ✓（必须分清 ✗）
    · 旧 coslot: 把字体渲染图当作**某位书家的替代品**注入（柳体→柳公权槽 ✓）
      ⇒ 目的是**补稀缺字** ✓，且注释明写"绝不新增任何 calligrapher_id" ✓ ⇒ **不新增槽位** ✗
    · 本工具: 让**每个字体各自成为一个槽位** ✓✓ ⇒ 判别器要**同时**区分 (书家) 与 (字体) ✓
      ⇒ 同字异风格对子从 ~2.4 涨到 ~24 ✓✓（书家 10 + 字体 ~17 ✓）
      ⇒ 印刷字应落到**字体槽**而非书家槽 ⇒ G3 从"负参考"升级为**可检验判据** ✓✓

渲染口径**必须复用仓库规范**（否则与既有数据不同分布 ✗）
    `tools/build_clean_v1.py::render_glyph` ✓ —— 256×256、白底黑字、墨迹 bbox 裁剪后
    缩放到 box_frac=0.88 并居中 ✓，且带**缺字(tofu)检测** ✓。

用法
    python tools/build_font_slot_dataset.py --max-chars 0 --out data/font_slots
    # 产出: <out>/imgs/<font>/<cp>.png + <out>/font_slots.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ★ 字体表: 刻意选**风格差异大**的 ✓（否则槽位之间学不出区分度 ✗）
FONT_TABLE = [
    # ★ 保留**实测可用**的仓库字体 ✓（其余 candidate_fonts 实测 0% 覆盖 ⇒ 可能是被裁剪/补丁过的 ✗）
    ("kai_simkai",      "simkai.ttf"),
    ("hei_simhei",      "SimHei.ttf"),
    ("li_simli",        "SIMLI.TTF"),
    ("xing_stxingka",   "STXINGKA.TTF"),
    # ★★ 系统字体（实测全部 10/10 覆盖 ✓，风格差异明显 ✓ ⇒ 16 个槽位 ✓✓）
    ("kai_simkai_win",  "C:/Windows/Fonts/simkai.ttf"),
    ("fang_simfang",    "C:/Windows/Fonts/simfang.ttf"),
    ("hei_simhei_win",  "C:/Windows/Fonts/simhei.ttf"),
    ("song_simsun",     "C:/Windows/Fonts/simsun.ttc"),
    ("yahei_msyh",      "C:/Windows/Fonts/msyh.ttc"),
    ("li_simli_win",    "C:/Windows/Fonts/SIMLI.TTF"),
    ("yuan_simyou",     "C:/Windows/Fonts/SIMYOU.TTF"),
    ("kai_stkaiti",     "C:/Windows/Fonts/STKAITI.TTF"),
    ("song_stsong",     "C:/Windows/Fonts/STSONG.TTF"),
    ("song_stzhongs",   "C:/Windows/Fonts/STZHONGS.TTF"),
    ("deng_deng",       "C:/Windows/Fonts/Deng.ttf"),
    ("deng_dengb",      "C:/Windows/Fonts/Dengb.ttf"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="exp-std/csv/train.csv",
                    help="取这里的**字符集合** ✓（与判别器要评的字一致 ✓）")
    ap.add_argument("--extra-csv", default="exp-std/csv/eval200_fixed.csv",
                    help="并入评测集的字 ✓（保证 G 系列有字可用 ✓）")
    ap.add_argument("--out", default="data/font_slots")
    ap.add_argument("--max-chars", type=int, default=0, help="0=全部 ✓")
    ap.add_argument("--min-freq", type=int, default=1,
                    help="只渲染**出现频次 ≥ 该值**的字 ✓（按频次 ✓ 不按 Unicode ✗）")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--verify-only", action="store_true", help="只检查字体可用性 ✓")
    args = ap.parse_args()
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
    from build_clean_v1 import render_glyph, FONT_DIR                # noqa: E402
    ROOT_ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # ★ 字体路径必须**自己解析** ✓（实测教训 ✓）
    #   `build_clean_v1.get_font()` 内部做 `os.path.join(FONT_DIR, fname)` ✓
    #   ⇒ 它要的是**相对 tools/fonts/ 的文件名**（如 "simkai.ttf"）✓；而**绝对路径**也行 ✓
    #     （os.path.join 遇绝对路径会直接返回它 ✓ —— 这就是为什么只有 C:/Windows 那两个能跑 ✓）。
    #   我第一版传 "tools/fonts/simkai.ttf" ✗ ⇒ 拼成 tools/fonts/tools/fonts/… ⇒ font=None
    #   ⇒ 被判为"缺字" ⇒ 16/17 字体报 0% 覆盖 ✗✗（虚惊一场 ✓，字体本身没问题 ✓）。
    def _resolve(fp: str) -> str:
        cands = [fp, os.path.join(FONT_DIR, fp), os.path.join(FONT_DIR, os.path.basename(fp)),
                 os.path.join(ROOT_, fp)]
        for c in cands:
            if os.path.isfile(c):
                # 若就在 FONT_DIR 里 ⇒ 传**裸文件名**（正合 get_font 的口径 ✓）
                if os.path.dirname(os.path.abspath(c)) == os.path.abspath(FONT_DIR):
                    return os.path.basename(c)
                return c.replace("\\", "/")
        return fp

    # ★ 字符要**按频次**收集 ✓（实测教训: 按 Unicode 排序取前 N 个会挑到大量**异体/繁体** ✗，
    #   小字库字体不收这些码位 ⇒ 会被误判为"覆盖 0%" ✗✗ —— 其实字体没问题 ✓。
    #   判据应是"**我们真正要评的字**覆盖多少" ✓ ⇒ 用行频次 ✓。）
    from collections import Counter
    freq = Counter()
    for p in (args.csv, args.extra_csv):
        if p and os.path.isfile(p):
            for r in csv.DictReader(open(p, encoding="utf-8-sig")):
                c = (r.get("character") or "").strip()
                if c:
                    freq[c] += 1
    chars = [c for c, n in freq.most_common() if n >= args.min_freq]  # 按频次 ✓
    chars_all = sorted(freq)                          # 全量（渲染时用 ✓）
    if args.max_chars:
        chars = chars[:args.max_chars]
    print("[fontslot] 字符 %d 个 | 字体 %d 个" % (len(chars), len(FONT_TABLE)), flush=True)

    # ① 字体可用性（按**覆盖率**判断 ✓ —— 覆盖太低的字体进不了槽位 ✓）
    ok_fonts, cover = [], {}
    for tag, fp in FONT_TABLE:
        # ★ 必须先 _resolve 再判存在性 ✗ —— 我上一版把解析放在 isfile **之后** ⇒
        #   16 个字体全报"缺文件" ✗（而文件其实都在 ✓，实测 Test-Path/Python isfile 均为 True ✓）
        fp = _resolve(fp)
        if not os.path.isfile(os.path.join(FONT_DIR, fp)) and not os.path.isfile(fp):
            print("  ✗ %-18s 缺文件 %s" % (tag, fp))
            continue
        probe = chars[:min(100, len(chars))]   # 最高频 100 个 ✓（我们真正要评的字 ✓）
        n = sum(1 for c in probe if render_glyph(c, fp, size=args.size) is not None)
        cover[tag] = n / max(1, len(probe))
        flag = "✓" if cover[tag] >= 0.6 else "✗(覆盖低)"
        print("  %s %-18s 覆盖 %.0f%% (%d/%d)" % (flag, tag, 100 * cover[tag], n, len(probe)))
        if cover[tag] >= 0.6:
            ok_fonts.append((tag, fp))
    print("[fontslot] 可用字体 %d 个 ✓" % len(ok_fonts))
    if args.verify_only:
        return 0
    if len(ok_fonts) < 4:
        print("  ✗ 可用字体不足 4 个 ⇒ 槽位设计无意义 ✓（请补字体 ✓）")
        return 3

    # ② 渲染（按字体分目录 ✓ 便于核对 ✓）
    rows, miss = [], defaultdict(int)
    for tag, fp in ok_fonts:
        d = os.path.join(args.out, "imgs", tag)
        os.makedirs(d, exist_ok=True)
        n_ok = 0
        for ch in chars_all:
            p = os.path.join(d, "%05x.png" % ord(ch))
            if os.path.isfile(p):
                n_ok += 1
            else:
                a = render_glyph(ch, fp, size=args.size)
                if a is None:
                    miss[tag] += 1
                    continue
                Image.fromarray(a).save(p)
                n_ok += 1
            rows.append({"path": p, "char": ch, "font": tag, "slot": "font:" + tag})
        print("  ✓ %-18s 渲染 %d / 缺字 %d" % (tag, n_ok, miss[tag]), flush=True)

    os.makedirs(args.out, exist_ok=True)
    csvp = os.path.join(args.out, "font_slots.csv")
    with open(csvp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "char", "font", "slot"])
        w.writeheader()
        w.writerows(rows)
    # 槽位索引表（供训练器把字体槽并进 pair 空间 ✓）
    with open(os.path.join(args.out, "font_slots_meta.json"), "w", encoding="utf-8") as f:
        import json
        json.dump({"fonts": [t for t, _ in ok_fonts],
                   "coverage": {t: cover[t] for t, _ in ok_fonts},
                   "n_chars": len(chars), "n_rows": len(rows),
                   "size": args.size}, f, ensure_ascii=False, indent=2)
    print("[fontslot] ✓ %d 行 -> %s" % (len(rows), csvp))
    print("  ⇒ 训练器用 --font-csv %s 即可把字体并成**独立槽位** ✓" % csvp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
