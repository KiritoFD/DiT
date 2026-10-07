# -*- coding: utf-8 -*-
"""eval_split_metrics.py — 固定三等分 (top/mid/worst) 上的 SSIM / LPIPS / IoU, 覆盖全部模型。

设计要点 (与仓库口径一致, 不另造一套):
  * SSIM  : src.eval.metrics.ssim      —— RGB 逐通道 + 高斯窗 (win=11, sigma=1.5)
  * LPIPS : src.eval.metrics.get_lpips —— **net="alex"** (与训练日志里的 lpips 同口径)
  * IoU   : src.eval.metrics_ink.ink_iou —— 墨迹 IoU (阈值由该模块定义)
  (骨架 IoU skel_iou 另算一列备用, 因为它的绝对值很小、量级不同)

三等分定义 (与 exp-std/csv/eval200_split_{top,mid,worst}.csv 完全一致):
  按 **v68 每样本 SSIM 降序**排, 62 / 63 / 62 三段 —— 即"v68 表现最好的 62 张"等。
  本脚本直接从 assets/eval200fix_models_per_sample.csv 重建, 并校验与已存 CSV 的行数一致。

图片: 根目录 eval/<model>/{i}.png, GT = eval/gt/{i}.png (索引 i = eval200_fixed.csv 行号)

用法:
    python tools/eval_split_metrics.py
    python tools/eval_split_metrics.py --models v68,v66,v54,moyi_12ch,moyi_4ch --device cpu
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

N = 187
GT_DIR = "eval/gt"

# 目录名 -> (前缀, 说明)。moyi_12ch 里有两套 (i.png 与 t{i}.png), 取 i.png。
MODELS = {
    "v70": ("", "ours Sp/2 + aug + stdskel + C2OT"),
    "v68": ("", "ours Sp/2 + aug + C2OT"),
    "v66": ("", "ours S/2 route2456"),
    "v54": ("", "ours S/2 三表 minimal"),
    # ── 48 机器容量阶梯 (2026-10-07 用**我们主线的 Heun 协议**在 48 上重新导出 187 张) ──
    # ⚠ 配方与我们不同: layer/gelu/qk_norm=0/rope=0/无REPA/无C2OT, lr 1.5e-4(B) / 1e-4(L)
    "v_b_aug_route_60k": ("", "48阶梯 B/2 130M · aug + route(2,4,5,6) · 60k"),
    "v_b_aug_40k": ("", "48阶梯 B/2 130M · aug · 40k"),
    "v_l_aug_route_50k": ("", "48阶梯 L/2 457M · aug + route(4,8,10,12) · 50k"),
    "v_sp_20k": ("", "48阶梯 Sp/2 59M · 原始数据 · 20k"),
    "v_b_base_5k": ("", "48阶梯 B/2 130M · 原始数据 · 5k (半训)"),
    "moyi_12ch": ("", "Moyi 官方 12ch (252M) 50k"),
    "moyi_4ch": ("", "Moyi 4ch (252M) 80k"),
    "v23": ("", "ours 旧臂 v23 (仅 38 张)"),
    "v21": ("", "ours 旧臂 v21 (仅 38 张)"),
    "v13": ("", "ours 旧臂 v13 (仅 38 张)"),
}


def build_splits():
    """按 v68 每样本 ssim 降序切 62/63/62; 再加第 4 档 **mid-strict** = mid ∩ 组合未见。

    mid-strict (用户 2026-10-07 裁定): 剔掉 (书家,书体,字) 组合在训练里出现过的样本。
    动机: mid 档 63 张里有 25 张 (40%) 的组合见过, 而该子集对**所有模型**都更容易
    (全 187 张上 v68 +0.0576 / v66 +0.0771, 见 tools/audit_eval_leakage2.py) -> 抬高绝对值。
    固定 CSV: exp-std/csv/eval200_split_midstrict.csv (由 tools/build_midstrict_split.py 生成)。
    """
    df = pd.read_csv("assets/eval200fix_models_per_sample.csv")
    d = df[df["model_name"] == "v68"].copy()
    d["ssim"] = d["ssim"].astype(float)
    d["idx"] = d["idx"].astype(int)
    d = d.sort_values("ssim", ascending=False).reset_index(drop=True)
    splits = {
        "top": set(d.iloc[0:62]["idx"].tolist()),
        "mid": set(d.iloc[62:125]["idx"].tolist()),
        "worst": set(d.iloc[125:187]["idx"].tolist()),
    }

    # ── mid-strict: 读固定 CSV, 并与 "mid ∩ 组合未见" 独立交叉核对 ──
    ids = {}
    with open("exp-std/csv/eval200_fixed.csv", encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f)):
            ids[r["image_path"]] = i
    K3 = ("calligrapher", "script", "character")
    seen3 = {tuple(str(r.get(k, "")).strip() for k in K3)
             for r in csv.DictReader(open("exp-std/csv/train.csv", encoding="utf-8"))}
    ev_rows = list(csv.DictReader(open("exp-std/csv/eval200_fixed.csv", encoding="utf-8")))
    un_idx = {i for i, r in enumerate(ev_rows)
              if tuple(str(r.get(k, "")).strip() for k in K3) not in seen3}
    midstrict = splits["mid"] & un_idx
    p = "exp-std/csv/eval200_split_midstrict.csv"
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as f:
            file_idx = {ids[r["image_path"]] for r in csv.DictReader(f)}
        flag = "✓" if file_idx == midstrict else "✗ 与独立重算不一致!"
        print(f"  [split] midstrict  固定CSV {len(file_idx):3d} 张 / 独立重算 mid∩未见 "
              f"{len(midstrict):3d} 张  {flag}")
        print(f"          (mid {len(splits['mid'])} 张 - 剔掉组合见过 "
              f"{len(splits['mid']) - len(midstrict)} 张 = {len(midstrict)} 张)")
    splits["midstrict"] = midstrict

    # 校验其余三档与已存 CSV 行数一致
    for name in ("top", "mid", "worst"):
        p = f"exp-std/csv/eval200_split_{name}.csv"
        if os.path.isfile(p):
            n = sum(1 for _ in open(p, encoding="utf-8")) - 1
            flag = "✓" if n == len(splits[name]) else "✗"
            print(f"  [split] {name:9s} 本脚本 {len(splits[name]):3d} 行 / 已存 CSV {n:3d} 行  {flag}")
    return splits


def load_rgb(p):
    with Image.open(p) as im:
        a = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="assets/eval200_splits_exact_metrics.csv")
    a = ap.parse_args()

    import torch
    from src.eval.metrics import ssim as our_ssim, get_lpips, skel_iou as our_skel_iou
    from src.eval.metrics_ink import ink_iou as our_ink_iou

    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    lp_fn, why = get_lpips(dev, net="alex")
    print(f"[lpips] net=alex device={dev} {'✓' if lp_fn is not None else '✗ ' + str(why)}")
    if lp_fn is None:
        return 1

    print("\n[split] 三等分 (按 v68 每样本 SSIM 降序, 62/63/62):")
    splits = build_splits()

    names = [m.strip() for m in a.models.split(",") if m.strip()]
    rows = []
    print(f"\n[run] {len(names)} 个模型 x {N} 张")
    for m in names:
        d = os.path.join("eval", m)
        if not os.path.isdir(d):
            print(f"  ⚠ 跳过 {m}: 目录不存在")
            continue
        pref = MODELS.get(m, ("", ""))[0]
        recs = []
        t0 = time.time()
        for i in range(N):
            pp = os.path.join(d, f"{pref}{i}.png")
            if not os.path.exists(pp):
                pp = os.path.join(d, f"{i}.png")
            gp = os.path.join(GT_DIR, f"{i}.png")
            if not (os.path.exists(pp) and os.path.exists(gp)):
                continue
            pr, gt = load_rgb(pp), load_rgb(gp)
            t_p = torch.from_numpy(pr.transpose(2, 0, 1))[None].to(dev) * 2 - 1
            t_g = torch.from_numpy(gt.transpose(2, 0, 1))[None].to(dev) * 2 - 1
            with torch.no_grad():
                lpv = float(lp_fn(t_p, t_g).mean().item())
            recs.append({"idx": i,
                         "ssim": float(our_ssim(pr, gt)),
                         "lpips": lpv,
                         "iou": float(our_ink_iou(pr, gt)),
                         "skel_iou": float(our_skel_iou(pr, gt, thresh=0.5))})
        df = pd.DataFrame(recs)
        print(f"  {m:10s} 算出 {len(df):3d}/{N} 张  ({time.time()-t0:.0f}s)")
        r = {"model": m, "n_total": len(df)}
        for sname, sidx in splits.items():
            sub = df[df["idx"].isin(sidx)]
            r[f"{sname}_n"] = len(sub)
            r[f"{sname}_ssim"] = sub["ssim"].mean() if len(sub) else float("nan")
            r[f"{sname}_lpips"] = sub["lpips"].mean() if len(sub) else float("nan")
            r[f"{sname}_iou"] = sub["iou"].mean() if len(sub) else float("nan")
            r[f"{sname}_skel_iou"] = sub["skel_iou"].mean() if len(sub) else float("nan")
        r["all_ssim"] = df["ssim"].mean()
        r["all_lpips"] = df["lpips"].mean()
        r["all_iou"] = df["iou"].mean()
        rows.append(r)

    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)

    # 打印 (统一写 ssim / lpips / iou)
    for sname, title in (("top", "TOP 62 (v68 表现最好的一档)"),
                         ("mid", "MID 63 (中等)"),
                         ("worst", "WORST 62 (最差的一档)")):
        print()
        print("=" * 96)
        print(f"### {title}   注: 每格都是**在该集合内**的均值  (~n={out[f'{sname}_n'].max()})")
        print(f"  {'model':12s} {'SSIM↑':>9s} {'LPIPS↓':>9s} {'IoU↑':>9s} "
              f"{'skelIoU↑':>10s} {'n':>5s}")
        print("  " + "-" * 88)
        sub = out.sort_values(f"{sname}_ssim", ascending=False)
        for _, r in sub.iterrows():
            print(f"  {r['model']:12s} {r[f'{sname}_ssim']:9.4f} {r[f'{sname}_lpips']:9.4f} "
                  f"{r[f'{sname}_iou']:9.4f} {r[f'{sname}_skel_iou']:10.5f} "
                  f"{int(r[f'{sname}_n']):5d}")

    print()
    print("=" * 96)
    print("### 全 187 张 (不切分, 供对照)")
    print(f"  {'model':12s} {'SSIM↑':>9s} {'LPIPS↓':>9s} {'IoU↑':>9s} {'n':>5s}")
    print("  " + "-" * 88)
    for _, r in out.sort_values("all_ssim", ascending=False).iterrows():
        print(f"  {r['model']:12s} {r['all_ssim']:9.4f} {r['all_lpips']:9.4f} "
              f"{r['all_iou']:9.4f} {int(r['n_total']):5d}")
    print(f"\n✓ 已保存 {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
