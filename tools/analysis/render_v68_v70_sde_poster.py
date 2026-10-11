#!/usr/bin/env python3
"""Render the v68/v70-family research poster from existing evidence; no inference.

--capture snapshots existing CSVs, aggregates paired statistics on CPU and records
source-image hashes. The default only renders the committed evidence snapshot.
Requires numpy and Pillow. Raw eval PNGs stay local; the poster is committed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/experiments/v68_v70_sde"
LOCAL = ROOT / "assets/results"
RANKS = [15, 45, 70, 90, 110, 135, 155, 175]
SEEDS = ["ode", "sde_g05", "sde_g05_seed1", "sde_g05_seed2"]
GRAY, INK, BLUE, ORANGE = "#667583", "#142a3b", "#176579", "#b85d28"


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bootstrap(delta, rng):
    means = np.empty(20000)
    for start in range(0, 20000, 1000):
        means[start:start + 1000] = delta[rng.integers(len(delta), size=(1000, len(delta)))].mean(1)
    return np.percentile(means, [2.5, 97.5]).tolist()


def image_rows():
    base = "assets/results/_local_v80_50k_{}/eval_samples_ctrl/step0050000/eval200fix/g{{idx}}.png"
    # The v79 55k batch is distinct from the stale eval/v79 directory.
    return [
        ("gt", "GT 真迹", "固定 eval200fix 的原图", "eval/gt/{idx}.png"),
        ("v68", "v68 · ODE", "200k · SD-VAE · aug + C2OT", "eval/v68/{idx}.png"),
        ("v68_sde", "v68 · 旧 SDE", "同 ckpt · γ0.5 · 有偏漂移", "eval/v68_sde/{idx}.png"),
        ("v70", "v70 · std-skel", "图像为 30k · 字表替换为骨架", "eval/v70/{idx}.png"),
        ("v72", "v72 · ODE", "130k · DINO-VAE · noaug + C2OT", "eval/v72/{idx}.png"),
        ("v72_sde", "v72 · 旧 SDE", "同 ckpt · γ0.5 · 结构收益未复现", "eval/v72_sde/{idx}.png"),
        ("v76", "v76 · topband", "40k · VAE 高频损失 · std≈0.20", "eval/v76/{idx}.png"),
        ("v77", "v77 · 去 REPA", "35k · 与 v76 图像不匹配步数", "eval/v77/{idx}.png"),
        ("v79", "v79 · ODE", "55k · 纯物理 VAE · std≈1 · C2OT", "eval/v79_55k/{idx}.png"),
        ("v79_sde", "v79 · 旧 SDE", "同 ckpt · γ0.5 · 连通性/高频权衡", "eval/v79_sde/{idx}.png"),
        ("v80", "v80 · ODE", "50k · 纯物理 VAE · no-OT", base.format("ode")),
        ("v80_sde", "v80 · corrected SDE", "同 ckpt · γ0.5 · 展示 seed0", base.format("sde_g05")),
    ]


def capture():
    OUT.mkdir(parents=True, exist_ok=True)
    original = read_csv(ROOT / "assets/eval200_splits_exact_metrics.csv")
    old_metrics = {r["model"]: {k: float(r[k]) for k in ["all_ssim", "all_lpips", "all_iou"]}
                   for r in original if r["model"] in {"v68", "v68_sde", "v72", "v72_sde"}}
    assert len(old_metrics) == 4
    meta = read_csv(ROOT / "exp-std/csv/eval200_fixed.csv")
    per_sample = read_csv(ROOT / "assets/eval200fix_models_per_sample.csv")
    ranked = sorted([r for r in per_sample if r["model_name"] == "v68"],
                    key=lambda r: (-float(r["ssim"]), int(r["idx"])))
    assert len(ranked) == 187
    selected = []
    for rank in RANKS:
        idx = int(ranked[rank]["idx"])
        selected.append({"idx": idx, "rank_zero_based": rank,
                         **{k: meta[idx][k] for k in ["character", "calligrapher", "script", "image_path"]}})
    paired = {}
    metrics = ["ssim", "mse", "lpips", "ink_ssim", "ink_iou", "skel_iou", "frag_ratio"]
    snapshot = []
    for name in SEEDS:
        path = LOCAL / f"_local_v80_50k_{name}/eval_stdskel_batch.csv"
        rows = read_csv(path)
        assert len(rows) == 187
        keyed = {int(r["idx"]): r for r in rows}
        assert set(keyed) == set(range(187)), f"Missing/duplicate paired IDs: {name}"
        for idx in range(187):
            assert keyed[idx]["char"] == meta[idx]["character"], (name, idx)
            assert int(keyed[idx]["step"]) == 50000 and keyed[idx]["set"] == "eval200fix"
            assert int(keyed[idx]["img_id"]) == int(Path(meta[idx]["image_path"]).stem)
        paired[name] = keyed
        snapshot.extend({"variant": name, **r} for r in rows)
    with (OUT / "v80_50k_paired.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(snapshot[0]))
        writer.writeheader()
        writer.writerows(snapshot)
    rng = np.random.default_rng(20261012)
    stats = {}
    for key in metrics:
        ode = np.array([float(paired["ode"][i][key]) for i in range(187)])
        paths = np.array([[float(paired[s][i][key]) for i in range(187)] for s in SEEDS[1:]])
        delta = paths.mean(0) - ode
        stats[key] = {"ode_mean": float(ode.mean()), "seed_means": paths.mean(1).tolist(),
                      "sde_mean": float(paths.mean()), "mean_delta": float(delta.mean()),
                      "ci95": bootstrap(delta, rng), "bootstrap_unit": "sample; averaged over 3 fixed path seeds"}
    paths = {}
    local_gt = LOCAL / "_local_v80_50k_ode/eval_samples_ctrl/step0050000/eval200fix"
    for sample in selected:
        idx = sample["idx"]
        with Image.open(ROOT / f"eval/gt/{idx}.png") as old_gt, Image.open(local_gt / f"gt{idx}.png") as current_gt:
            assert np.array_equal(np.asarray(old_gt.convert("RGB")), np.asarray(current_gt.convert("RGB"))), f"GT differs: {idx}"
    for row_id, _, _, pattern in image_rows():
        for sample in selected:
            path = ROOT / pattern.format(idx=sample["idx"])
            if not path.is_file():
                raise FileNotFoundError(path)
            with Image.open(path) as im:
                assert im.size == (256, 256), (path, im.size)
            paths[path.relative_to(ROOT).as_posix()] = sha(path)
    points = read_csv(ROOT / "assets/ledger/points_all.csv")
    curve = [r for r in points if r["run"] in {"v68_aug_sp_c2ot", "v70_aug_sp_stdskel_c2ot"}
             and r["set"] == "eval200fix" and int(r["step"]) in {30000, 65000}]
    assert len(curve) == 4
    evidence = {
        "date": "2026-10-12", "n": 187, "cfg": 1.0, "steps": 50,
        "note": "Historical image metrics, historical scans and v80 batch metrics are separate streams.",
        "selection": "v68 SSIM rank, fixed zero-based ranks; no SDE delta selection",
        "samples": selected, "old_image_metrics": old_metrics,
        "historical_reported": {
            "v68": {"frag": 5.06, "hf_gt": .811}, "v68_sde": {"frag": 2.38, "hf_gt": .760},
            "v72": {"frag": 2.566, "hf_gt": .622}, "v72_sde": {"frag": 2.867, "hf_gt": .630},
            "v76": {"hf_gt": .629}, "v77": {"hf_gt": .623},
            "v79": {"ssim": .5799, "lpips_alex": .4318, "ink_iou": .2719, "frag": 1.876, "hf_gt": .647},
            "v79_sde": {"ssim": .5788, "lpips_alex": .4283, "ink_iou": .2838, "frag": 1.466, "hf_gt": .622},
            "source": "docs/sde_analysis_brief.md sections 3.1 and 3.3; rounded historical readouts, not recomputed"
        },
        "v68_v70_matched_curve": curve, "v80_three_seed": stats,
        "legacy_sweep": {
            "cfgs": [.4, .6, .7, .8, 1., 1.2], "gammas": [0., .2, .35, .5],
            "v68_ssim": [[.5615, .5727, .5911, .6125], [.5985, .6052, .6161, .6282],
                          [.6088, .6136, .6208, .6289], [.6132, .6166, .6217, .6281],
                          [.6150, .6172, .6206, .6253], [.6141, .6161, .6190, .6223]],
            "v72_ssim": [[float(r["ssim"]) for r in read_csv(ROOT / "assets/ledger/v72_sweep_130k_2d.csv")
                          if float(r["cfg"]) == cfg] for cfg in [.4, .6, .7, .8, 1., 1.2]],
            "v68_source": "docs/experiments/2026-10-08-sit-sde-sampler-investigation-and-sweep.md section 6.1",
            "v72_source": "assets/ledger/v72_sweep_130k_2d.csv",
            "note": "Legacy drift; separate scan batches, not the poster image batches. LPIPS not mixed."
        },
        "image_sha256": paths,
        "source_sha256": {p: sha(ROOT / p) for p in [
            "assets/eval200_splits_exact_metrics.csv", "assets/ledger/points_all.csv",
            "assets/ledger/v72_sweep_130k_2d.csv", "assets/eval200fix_models_per_sample.csv",
            "exp-std/csv/eval200_fixed.csv", "docs/sde_analysis_brief.md",
            "docs/experiments/2026-10-08-sit-sde-sampler-investigation-and-sweep.md",
            *[f"assets/results/_local_v80_50k_{name}/eval_stdskel_batch.csv" for name in SEEDS]]},
    }
    (OUT / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Captured existing evidence; no model/VAE loaded.")


def font(size, bold=False):
    for p in [Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
              Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")]:
        if p.is_file():
            return ImageFont.truetype(str(p), size)
    raise RuntimeError("Install a Chinese font (Microsoft YaHei or Noto Sans CJK).")


def render(verify=True):
    evidence = json.loads((OUT / "evidence.json").read_text(encoding="utf-8"))
    width, height = 3140, 5830
    im = Image.new("RGB", (width, height), "#f7f6f2")
    d = ImageDraw.Draw(im)

    def text(x, y, value, size=27, color=INK, bold=False):
        value = str(value)
        box = d.textbbox((x, y), value, font=font(size, bold))
        if box[2] > width - 25:
            raise ValueError(f"Text exceeds poster width: {value}")
        d.text((x, y), value, font=font(size, bold), fill=color)

    def lines(x, y, values, size=26, color=INK, step=43):
        for j, v in enumerate(values):
            text(x, y + j * step, v, size, color)

    def section(y, tag, title):
        d.line((50, y, width - 50, y), fill="#d3d9d9", width=2)
        text(50, y + 15, tag, 25, BLUE, True)
        text(145, y + 8, title, 35, INK, True)

    d.rectangle((0, 0, width, 210), fill=INK)
    text(50, 24, "v68 → v70 系列：SDE 修了什么，什么时候还能获益？", 58, "#ffffff", True)
    text(52, 109, "历史消融 / 同字图像 / 旧式与 corrected SDE / 新模型的推理收益空间", 32, "#dce8e9")
    text(52, 158, "证据截止 2026-10-12  ·  eval200fix N=187  ·  展示统一 CFG=1 / Heun-50  ·  本页由已有产物生成", 25, "#dce8e9")

    section(240, "01", "实验链路：训练消融与采样消融分别读")
    xcols = [50, 440, 790, 1450, 2140]
    heads = ["实验", "已有结果/状态", "主要变化", "能读到的结论", "比较边界"]
    for x, h in zip(xcols, heads):
        text(x, 305, h, 25, GRAY, True)
    lineage = [
        ["v68", "200k 基准", "SD-VAE / aug / C2OT / 三表", "旧 SDE 显著修碎笔", "SSIM↑，LPIPS/ink IoU↓"],
        ["v70", "停于 78.7k", "字表 → VAE 编码 std-skel", "65k SSIM .5535 vs v68 .5833", "图像仅 30k；不是最终图"],
        ["v71 / v73 / v75", "作废 / 未启动 / 停止", "旧 latent 契约 / REPA-E / DoG", "不提供有效的 SDE 对照", "缺有效结果，不补造数据"],
        ["v72 → v74", "130k / mode 消融", "DINO-VAE + noaug / sample→mode", "mode：23/23 点落后", "v68→v72 同时变语料+VAE"],
        ["v76 → v77 → v78", "topband / 去 REPA / scale", "VAE 高频 / w_repa=0 / std≈1", "看各自配对训练曲线", "图像 40k/35k；v78 未列图"],
        ["v79 → v80", "55k → 50k", "纯物理 VAE / C2OT→no-OT", "v80 corrected：frag 下降", "只在各 ckpt 内做 ODE/SDE A/B"],
    ]
    for j, row in enumerate(lineage):
        yy = 350 + 53 * j
        for x, value in zip(xcols, row):
            text(x, yy, value, 25)

    section(690, "02", "SDE 的实测作用：同一 checkpoint 内比较，避免把训练进展当成采样收益")
    columns = [50, 750, 1230, 1710, 2220]
    for x, label in zip(columns, ["配对实验 / 数据流", "SSIM ↑", "LPIPS-alex ↓", "frag 生成/GT 连通块比", "HF/GT · 描述性频谱"]):
        text(x, 750, label, 25, GRAY, True)
    a = evidence["old_image_metrics"]
    table = [
        ["v68 200k · 旧 SDE · 历史图像批", f"{a['v68']['all_ssim']:.4f} → {a['v68_sde']['all_ssim']:.4f}",
         f"{a['v68']['all_lpips']:.4f} → {a['v68_sde']['all_lpips']:.4f}", "5.06 → 2.38*", ".811 → .760*"],
        ["v72 130k · 旧 SDE · 历史图像批", f"{a['v72']['all_ssim']:.4f} → {a['v72_sde']['all_ssim']:.4f}",
         f"{a['v72']['all_lpips']:.4f} → {a['v72_sde']['all_lpips']:.4f}", "2.566 → 2.867*", ".622 → .630*"],
        ["v79 55k · 旧 SDE · 历史简报", ".5799 → .5788*", ".4318 → .4283*", "1.876 → 1.466*", ".647 → .622*"],
        ["v80 50k · corrected · 3 seed 均值",
         f"{evidence['v80_three_seed']['ssim']['ode_mean']:.6f} → {evidence['v80_three_seed']['ssim']['sde_mean']:.6f}",
         f"{evidence['v80_three_seed']['lpips']['ode_mean']:.6f} → {evidence['v80_three_seed']['lpips']['sde_mean']:.6f}",
         f"{evidence['v80_three_seed']['frag_ratio']['ode_mean']:.6f} → {evidence['v80_three_seed']['frag_ratio']['sde_mean']:.6f}",
         "seed0: .633 → .638"],
    ]
    for j, row in enumerate(table):
        yy = 798 + 58 * j
        for x, value in zip(columns, row):
            text(x, yy, value, 26, BLUE if j == 3 else INK, j == 3)
    s, f = evidence["v80_three_seed"]["ssim"], evidence["v80_three_seed"]["frag_ratio"]
    text(50, 1046, f"v80：ΔSSIM {s['mean_delta']:+.6f}，95% CI [{s['ci95'][0]:+.6f}, {s['ci95'][1]:+.6f}]；"
         f"Δfrag {f['mean_delta']:+.5f}，CI [{f['ci95'][0]:+.5f}, {f['ci95'][1]:+.5f}]。", 28, BLUE, True)
    text(50, 1093, "* 历史简报读数；不同数据流只读组内变化。旧式漂移+C2OT 无边缘保持保证；frag 更低也可能来自并笔/丢笔。", 25, GRAY)

    section(1150, "03", "旧 SDE 的 γ × CFG 变体：历史扫描 SSIM；两批扫描仅各自读行内变化")
    sweep = evidence["legacy_sweep"]
    for xx, key, title in [(50, "v68_ssim", "v68 @200k · 旧漂移：低 CFG 时修复空间更大"),
                           (1590, "v72_ssim", "v72 @130k · 旧漂移：γ0.5 在 CFG≥0.6 反而下降")]:
        text(xx, 1214, title, 29, INK, True)
        text(xx, 1264, "CFG / γ", 25, GRAY)
        for col, gamma in enumerate(sweep["gammas"]):
            text(xx + 250 + 270 * col, 1264, f"γ={gamma:.2f}", 25, GRAY)
        for row, cfg in enumerate(sweep["cfgs"]):
            yy = 1307 + row * 34
            values = sweep[key][row]
            text(xx, yy, f"{cfg:.2f}", 25, GRAY)
            for col, value in enumerate(values):
                text(xx + 250 + 270 * col, yy, f"{value:.4f}", 25,
                     BLUE if col == int(np.argmax(values)) else INK, col == int(np.argmax(values)))
    text(50, 1538, "注意：v68 扫描 γ0.5 的 SSIM 每行最高，但 LPIPS 并非每行最优；不能称为“全指标免费午餐”。", 26, ORANGE)

    section(1610, "04", "同字图像：8 个固定 v68 排名位置，所有行共享列；按实验关系排序")
    text(50, 1673, "零基排名 15 / 45 / 70 / 90 / 110 / 135 / 155 / 175；按 v68 定义难度，因此跨档均值有选择偏差。", 25, GRAY)
    left, cell, gap, top, row_h = 870, 256, 20, 1850, 270
    for j, s in enumerate(evidence["samples"]):
        xx = left + j * (cell + gap)
        text(xx, 1725, f"{s['character']} · {s['script']}", 30, INK, True)
        text(xx, 1770, s["calligrapher"], 25, GRAY)
        text(xx, 1805, f"idx {s['idx']} · rank {s['rank_zero_based']}", 21, GRAY)
    for i, (key, title, desc, pattern) in enumerate(image_rows()):
        yy = top + i * row_h
        text(50, yy + 30, title, 35, BLUE if key.startswith("v80") else INK, True)
        text(50, yy + 90, desc, 27, GRAY)
        notes = {
            "gt": "对照图像；原始 256×256，无锐化、无反色",
            "v68": "基线保留飞白，但部分笔画断成碎片",
            "v68_sde": "SSIM +.00915；ink IoU −.00838",
            "v70": "65k 曲线落后；30k 图像不能代表 78.7k",
            "v72": "潜空间与语料一起改变；不是纯 VAE 消融",
            "v72_sde": "SSIM −.00117；frag 未改善",
            "v76": "历史 HF/GT .629；只作图像观察",
            "v77": "历史 HF/GT .623；不能与 40k 归因比较",
            "v79": "使用 eval/v79_55k，排除旧 eval/v79 批",
            "v79_sde": "连通性改善，同时高频减少",
            "v80": "本地 ODE .583217 ≈ 4090 .5835",
            "v80_sde": "3 seed frag 同向下降；SSIM 增益未显著",
        }
        text(50, yy + 143, notes[key], 25, GRAY)
        for j, s in enumerate(evidence["samples"]):
            path = ROOT / pattern.format(idx=s["idx"])
            rel = path.relative_to(ROOT).as_posix()
            if verify and sha(path) != evidence["image_sha256"][rel]:
                raise RuntimeError(f"Image changed since evidence capture: {rel}")
            with Image.open(path) as source:
                assert source.size == (256, 256)
                im.paste(source.convert("RGB"), (left + j * (cell + gap), yy))

    section(5110, "05", "能否有 free lunch？有推理收益空间；关键在扩散的时段和可靠的条件场")
    # Reserve the full mathematical explanation for the companion document.
    lines(50, 5170, [
        "corrected: b=v−D(t)·score，g²=2D；score=−[x+(1−t)v]/t；代码 dt<0。精确场同分布，实际场可以获益。",
        "旧式: b=(1+γ²t)v；有偏外推，不保证边缘保持。v68 的大涨是真实组内观测，旧漂移与 C2OT 的贡献尚未分离。",
    ], size=28, step=48)
    for x, title, body in [
        (50, "成立条件", ["线性训练路径，ε独立于x0（给定条件）", "推理时关OT不能修复旧C2OT训练耦合", "时间D与score漂移必须同步调整", "严格对照先CFG1；随机Heun共享dW", "std≈1有利优化，但不是公式成立条件"]),
        (1080, "首选未测候选", ["D(t)=γ₀² t(1−t) w(t)²", "前段布局 → 中段弱SDE → 末段ODE", "w(t)平滑时间窗；γ₀先考虑.2/.35", "理由：当前桥式末段放大场误差", "代价：不重训、保持Heun NFE100"]),
        (2110, "怎样才算 free lunch", ["验证集选日程，独立测试集确认", "多个起始seed + 多个路径seed", "SSIM/LPIPS/字形/飞白共同检查", "frag降低要排除并笔、丢笔", "当前只证小幅碎片率收益，非全质量提升"]),
    ]:
        text(x, 5322, title, 34, BLUE, True)
        lines(x, 5380, body, size=27, step=52)
    text(50, 5680, "参考：SiT §2.4/3.2（训练后可调扩散系数）；Cao et al. 2023（误差时段影响ODE/SDE优劣）；迁移效果未验证。", 26, GRAY)
    text(50, 5730, "完整推导、证据边界与复现：docs/experiments/2026-10-12-v68-v70-sde-role-and-free-lunch.md", 26, GRAY)
    text(50, 5770, "此页只聚合已有数据；evidence.json记录96张源图的SHA-256；原尺寸图像无重采样；poster不是训练终点排行榜。", 25, GRAY)
    OUT.mkdir(parents=True, exist_ok=True)
    im.save(OUT / "poster.png", optimize=True)
    preview = im.copy()
    preview.thumbnail((1570, 2500), Image.Resampling.LANCZOS)
    preview.save(OUT / "poster_preview.jpg", quality=92, subsampling=0)
    print(f"Rendered {OUT / 'poster.png'} ({width}x{height})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture", action="store_true", help="Snapshot existing artifacts; never runs inference")
    args = ap.parse_args()
    if args.capture:
        capture()
    render()


if __name__ == "__main__":
    main()
