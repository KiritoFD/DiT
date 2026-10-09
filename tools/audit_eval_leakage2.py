# -*- coding: utf-8 -*-
"""audit_eval_leakage2.py — 第二级审计: "组合见过"到底会不会污染 Best/Mid/Worst 档。

第一级已确认: 三个**样本级**主键 (img_id / image_path / src_image_path) 与 train.csv **零重叠**,
即没有"同一张图"进训练。但 (书家,书体,字) 组合有 75/187 命中 —— 也就是"这个书家用这个书体写这个字"
这件事模型见过 (只是不同的那一张)。

本脚本回答两个真正决定海报可读性的问题:
  Q1 这 75 个"组合见过"的样本, 落在 Best / Mid / Worst 三档的分布是否倾斜?
     (若大量堆在 Best 档 -> Best 海报的高分有相当部分是"组合记忆", 不能当泛化读)
  Q2 **在同一档内部**, "组合见过"子集 与 "组合未见"子集 的 v68 SSIM 差多少?
     (这是最干净的对照: 档位已固定, 只比子集)
"""
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EVAL = "exp-std/csv/eval200_fixed.csv"
TRAIN = "exp-std/csv/train.csv"
SPLITS = {"best": "exp-std/csv/eval200_split_top.csv",
          "mid": "exp-std/csv/eval200_split_mid.csv",
          "worst": "exp-std/csv/eval200_split_worst.csv"}
PER_SAMPLE = "assets/eval200fix_models_per_sample.csv"


def rows(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def tup(r, keys):
    return tuple(str(r.get(k, "")).strip() for k in keys)


def main():
    ev = rows(EVAL)
    tr = rows(TRAIN)
    idx_of = {r["image_path"]: i for i, r in enumerate(ev)}

    # 训练集里见过的 组合 / (书家,字) / 字
    K3 = ("calligrapher", "script", "character")
    K2 = ("calligrapher", "character")
    seen3 = {tup(r, K3) for r in tr}
    seen2 = {tup(r, K2) for r in tr}
    seen1 = {str(r["character"]).strip() for r in tr}

    flag = {}
    for i, r in enumerate(ev):
        flag[i] = (tup(r, K3) in seen3, tup(r, K2) in seen2,
                   str(r["character"]).strip() in seen1)

    n3 = sum(1 for f in flag.values() if f[0])
    n2 = sum(1 for f in flag.values() if f[1])
    n1 = sum(1 for f in flag.values() if f[2])
    print(f"[eval] 187 行中: (书家,书体,字) 见过 {n3} / (书家,字) 见过 {n2} / 仅字见过 {n1}")
    print(f"[eval] 不同字 {len({r['character'] for r in ev})} 个, "
          f"其中在训练字表里 {len({r['character'] for r in ev} & seen1)} 个")
    print(f"[eval] 样本级 (img_id/image_path) 与训练重叠: 0 (见 audit_eval_leakage.py)")

    # 档位
    tier_map = {}
    for t, p in SPLITS.items():
        for r in rows(p):
            tier_map[idx_of[r["image_path"]]] = t

    # 每样本 v68 SSIM
    ps = {int(r["idx"]): float(r["ssim"]) for r in rows(PER_SAMPLE)
          if r["model_name"] == "v68"}

    print()
    print("=" * 96)
    print("Q1: '组合见过' 在三档里的分布")
    print(f"  {'档':6s} {'n':>4s} {'组合见过':>8s} {'占比':>7s} {'组合见过均值SSIM':>16s} {'组合未见均值SSIM':>16s} {'Δ':>9s}")
    for t in ("best", "mid", "worst"):
        ids = [i for i, tt in tier_map.items() if tt == t]
        a = [ps[i] for i in ids if flag[i][0]]
        b = [ps[i] for i in ids if not flag[i][0]]
        ma = sum(a) / len(a) if a else float("nan")
        mb = sum(b) / len(b) if b else float("nan")
        print(f"  {t:6s} {len(ids):4d} {len(a):8d} {len(a)/max(1,len(ids))*100:6.1f}% "
              f"{ma:16.4f} {mb:16.4f} {ma-mb:9.4f}")

    print()
    print("=" * 96)
    print("Q2: 同一档内部, '组合见过' vs '组合未见' 的 v68 SSIM 对照 (最干净)")
    for t in ("best", "mid", "worst"):
        ids = [i for i, tt in tier_map.items() if tt == t]
        for nm, key in (("(书家,书体,字)", 0), ("(书家,字)", 1), ("字", 2)):
            a = [ps[i] for i in ids if flag[i][key]]
            b = [ps[i] for i in ids if not flag[i][key]]
            if not a or not b:
                continue
            ma, mb = sum(a) / len(a), sum(b) / len(b)
            print(f"  {t:6s} 见过{nm:12s} n={len(a):3d} SSIM={ma:.4f}  |  "
                  f"未见 n={len(b):3d} SSIM={mb:.4f}  |  Δ={ma-mb:+.4f}")

    print()
    print("=" * 96)
    print("Q3: 全 187 张上, 组合见过 vs 未见的整体差")
    a = [ps[i] for i in flag if flag[i][0]]
    b = [ps[i] for i in flag if not flag[i][0]]
    print(f"  组合见过 n={len(a)} SSIM={sum(a)/len(a):.4f}   "
          f"组合未见 n={len(b)} SSIM={sum(b)/len(b):.4f}   "
          f"Δ={sum(a)/len(a)-sum(b)/len(b):+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
