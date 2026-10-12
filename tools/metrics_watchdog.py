#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""metrics_watchdog.py — **纯 CPU watchdog**：盯目录、拿已落盘的图算指标、写 metrics.json。

架构（真异步，GPU 永不等指标）:
    ┌─ GPU 进程: 采样 -> 解码 -> 落盘 g{i}.png / gt{i}.png ─┐   立刻推下一个 ckpt ✓
    │                                                      │
    └─ 本进程(可开 N 个): 轮询 step*/ 目录 -> 图片齐了就算指标 -> metrics.json ✓

为什么可行: 评测图像**本来就是落盘的**（`eval_samples_ctrl/step*/<set>/g{i}.png` + `gt{i}.png`），
所以指标计算不需要模型、不需要 VAE、不需要 GPU，也不占训练显存 ✓。

数值口径: **指标全部从磁盘 PNG 计算** —— 与"内存里算"相比只差 8bit 量化，
但只要所有 ckpt 都走这条路，横向就是同一把尺 ✓（这也是可复现性最好的做法 ✓）。

用法:
  PY=/opt/conda/envs/cu121/bin/python
  # 单次巡检（把没算过的都算掉，遇到没出完图的跳过）
  $PY tools/metrics_watchdog.py --root assets/results/v83A_sweep_correct_20261012 --vae-path <vae> --once
  # 常驻：每 20 秒扫一次（GPU 出新目录就自动跟上）
  $PY tools/metrics_watchdog.py --root <dir> --vae-path <vae> --interval 20 --threads 32
  # 自检：与已有 metrics.json 对拍（验证本脚本口径）
  $PY tools/metrics_watchdog.py --root <dir> --selfcheck assets/results/<ref>/step0035000/metrics.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.isdir(os.path.join(ROOT, "src")):
    os.chdir(ROOT)
sys.path.insert(0, os.getcwd())

# ── 指标一律**复用** in_mem_eval / metrics 里的实现，绝不另写一套（防口径漂移 ✗）──
from src.eval.metrics_ink import ink_iou as _ink_iou, ink_ssim as _ink_ssim          # noqa: E402
from src.eval.metrics import skel_iou as _skel_iou, frag_ratio as _frag, hole_ratio as _hole   # noqa: E402
from src.eval.in_mem_eval import _lpips_per_sample                                    # noqa: E402

_GRAY = {}


def load_img(path):
    """PNG -> (H,W,3) float32 [0,1]（与内存口径一致: 0..1 而非 0..255）✓"""
    a = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    return a


def ssim_default(a, b):
    from skimage.metrics import structural_similarity as _ss
    return float(_ss(a, b, channel_axis=2, data_range=1.0))


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def soft_mass(a):
    return float(np.mean(1.0 - a.mean(axis=2)))


def _sets_from_manifest(sd):
    """从评测器写的 manifest.json 取 set 顺序与 n（权威切分依据）。

    in_mem_eval 把多个 set 的图**混在同一个目录**里（step*/g/g{i}.png，
    i 按 set 顺序连续编号），所以只能靠 manifest 的 caches 顺序 + n 切分 ✓。
    返回 [(set_name, start, n)]；没有 manifest 时返回 None（退化为整目录一个 set）。
    """
    mp = os.path.join(sd, "manifest.json")
    if not os.path.isfile(mp):
        return None
    try:
        man = json.load(open(mp, encoding="utf-8"))
        caches = man.get("caches") or {}
        out, start = [], 0
        for name, info in caches.items():
            n = int(info.get("n") or 0)
            if n > 0:
                out.append((str(name), start, n))
                start += n
        return out or None
    except Exception as e:                                        # noqa: BLE001
        print("[wd] manifest 解析失败 %s: %r" % (sd, e))
        return None


def collect(root, skip_done=True, only_step=None):
    """返回 [(step, set_name, [g...], [gt...], step_dir)]，只挑"图齐且未算完"的。"""
    out = []
    # ??? step*??????? <root>/eval_samples_ctrl/step*/...?????????????
    for sd in sorted(glob.glob(os.path.join(root, "**", "step*"), recursive=True)):
        if not os.path.isdir(sd):
            continue
        step = os.path.basename(sd).replace("step", "")
        if only_step and step.lstrip("0") != str(only_step).lstrip("0"):
            continue
        if skip_done and os.path.exists(os.path.join(sd, "metrics.json")):
            continue
        # 同一目录内配对（真实布局 eval_samples_ctrl/<step>/g/{g,gt}*.png）
        gtdirs = sorted({os.path.dirname(p) for p in
                         glob.glob(os.path.join(sd, "**", "gt[0-9]*.png"), recursive=True)})
        for gtdir in gtdirs:
            gs = sorted(glob.glob(os.path.join(gtdir, "g[0-9]*.png")),
                        key=lambda p: int(os.path.basename(p)[1:-4]))
            gts = sorted(glob.glob(os.path.join(gtdir, "gt[0-9]*.png")),
                         key=lambda p: int(os.path.basename(p)[2:-4]))
            if not gs or len(gs) != len(gts):
                print("[wd] %s: g=%d gt=%d 未齐，等待" % (step, len(gs), len(gts)))
                continue
            splits = _sets_from_manifest(sd)
            if splits:
                for name, start, n in splits:
                    if start + n > len(gs):
                        print("[wd] %s/%s 需要 %d 张但只有 %d，等待"
                              % (step, name, start + n, len(gs)))
                        continue
                    out.append((step, name, gs[start:start + n], gts[start:start + n], sd))
            else:
                out.append((step, "n%d" % len(gs), gs, gts, sd))
    return out


def evaluate_one(job, lpips_on=True, threads=16):
    step, setname, gs, gts, sd = job
    n = len(gs)
    pred = np.stack([load_img(p) for p in gs])
    gt = np.stack([load_img(p) for p in gts])

    def per_image(i):
        return dict(
            mse=mse(pred[i], gt[i]),
            ssim=ssim_default(pred[i], gt[i]),
            ink_ssim=float(_ink_ssim(pred[i], gt[i])),
            ink_iou=float(_ink_iou(pred[i], gt[i])),
            skel_iou=float(_skel_iou(pred[i], gt[i], thresh=0.5)),
            frag=float(_frag(pred[i], gt[i], thresh=0.5)),
            hole=float(_hole(pred[i], thresh=0.5)),
            hole_gt=float(_hole(gt[i], thresh=0.5)),
        )

    with ThreadPoolExecutor(max_workers=max(1, threads)) as ex:
        rows = list(ex.map(per_image, range(n)))

    lp = _lpips_per_sample(pred, gt, enabled=lpips_on) or [None] * n
    sm_p = np.array([soft_mass(pred[i]) for i in range(n)], dtype=np.float64)
    sm_g = np.array([soft_mass(gt[i]) for i in range(n)], dtype=np.float64)
    ratio = sm_p / np.maximum(sm_g, 1e-6)

    def mean(k):
        v = [r[k] for r in rows if r.get(k) is not None]
        return float(np.mean(v)) if v else None

    res = {
        setname: {
            "n": n,
            "ssim": mean("ssim"), "mse": mean("mse"),
            "lpips": (float(np.mean([x for x in lp if x is not None])) if any(x is not None for x in lp) else None),
            "ink_ssim": mean("ink_ssim"), "ink_iou": mean("ink_iou"),
            "skel_iou": mean("skel_iou"), "frag": mean("frag"),
            "hole": mean("hole"), "hole_gt": mean("hole_gt"),
            # ★ 白化判据（v81 事故里 ssim/lpips/ink_iou 全看不出来 ✗）
            "soft_ink_ratio": float(np.mean(ratio)),
            "near_blank_rate": float(np.mean((ratio < 0.10).astype(np.float64))),
            "gt_soft_mass": float(np.mean(sm_g)),
            "_source": "metrics_watchdog (from disk PNG)",
        }
    }
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--interval", type=float, default=0.0, help=">0 则常驻轮询")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--no-lpips", action="store_true")
    ap.add_argument("--selfcheck", default=None, help="与给定 metrics.json 对拍后退出")
    ap.add_argument("--selfcheck-step", default=None, help="自检指定的 step（默认第一个）")
    a = ap.parse_args()
    try:
        import torch
        torch.set_num_threads(max(4, a.threads))
    except Exception:                                             # noqa: BLE001
        pass

    if a.selfcheck:
        jobs = collect(a.root, skip_done=False, only_step=a.selfcheck_step)
        if not jobs:
            print("[wd] 没有可对拍的任务"); return
        got = evaluate_one(jobs[0], lpips_on=not a.no_lpips, threads=a.threads)
        ref = json.load(open(a.selfcheck, encoding="utf-8"))
        print("[wd] 自检对拍（磁盘 PNG vs 内存参考）:")
        for s in sorted(set(got) & set(ref)):
            for k in ("ssim", "lpips", "ink_iou", "frag", "hole", "soft_ink_ratio", "near_blank_rate"):
                o, n = ref[s].get(k), got[s].get(k)
                if isinstance(o, (int, float)) and isinstance(n, (int, float)):
                    print("   %-11s %-15s ref=%-10.6f wd=%-10.6f  Δ=%+.2e" % (s, k, o, n, n - o))
        return

    rounds = 0
    while True:
        rounds += 1
        jobs = collect(a.root)
        if jobs:
            for job in jobs:
                step, setname = job[0], job[1]
                t0 = time.time()
                res = evaluate_one(job, lpips_on=not a.no_lpips, threads=a.threads)
                out = os.path.join(job[4], "metrics.json")
                # 同一 step 多个 set 时合并写
                if os.path.exists(out):
                    old = json.load(open(out, encoding="utf-8"))
                    old.update(res)
                    res = old
                json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
                m = res[setname]
                print("[wd] step=%s set=%-11s n=%d ssim=%.4f lpips=%s inkIoU=%.4f "
                      "softInk=%.3f blank=%.4f (%.0fs)"
                      % (step, setname, m["n"], m["ssim"] or 0,
                         ("%.4f" % m["lpips"]) if m["lpips"] is not None else "NA",
                         m["ink_iou"] or 0, m["soft_ink_ratio"], m["near_blank_rate"], time.time() - t0),
                      flush=True)
        else:
            print("[wd] 第 %d 轮: 无新图" % rounds, flush=True)
        if a.once or a.interval <= 0:
            break
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
