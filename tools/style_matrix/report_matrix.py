#!/usr/bin/env python3
"""Offline S/J matrix reporting without touching a training GPU.

The command has three small jobs:

* ``summarize``: bootstrap the existing per-sample evaluation CSVs;
* ``geometry``: score paired generated/GT PNGs with interpretable geometry;
* ``poster``: render a compact cross-arm comparison from existing posters.

All random choices are deterministic.  The tool never loads a generator or VAE.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


SEED = 20261028
METRICS = {
    "ssim": True,
    "lpips": False,
    "ink_iou": True,
    "skel_iou": True,
    "frag_ratio": False,
    "soft_ink_ratio": None,
}


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _finite(values: Iterable[float]) -> np.ndarray:
    a = np.asarray(list(values), np.float64)
    return a[np.isfinite(a)]


def _mean_ci(values: Iterable[float], n_boot: int = 2000) -> dict[str, float | int]:
    a = _finite(values)
    if not len(a):
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(SEED)
    means = np.empty(n_boot, np.float64)
    for start in range(0, n_boot, 200):
        take = min(200, n_boot - start)
        idx = rng.integers(0, len(a), size=(take, len(a)))
        means[start:start + take] = a[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": float(a.mean()), "lo": float(lo), "hi": float(hi), "n": int(len(a))}


def _rate_ci(mask: Iterable[bool]) -> dict[str, float | int]:
    return _mean_ci(np.asarray(list(mask), np.float64))


def _arm_summary(rows: list[dict[str, str]]) -> dict:
    out = {"n": len(rows), "metrics": {}, "failure_rates": {}, "by_writer": {}}
    for key in METRICS:
        out["metrics"][key] = _mean_ci(float(r[key]) for r in rows)
    out["metrics"]["soft_ink_abs_error"] = _mean_ci(
        abs(float(r["soft_ink_ratio"]) - 1.0) for r in rows)
    out["metrics"]["hole_abs_error"] = _mean_ci(
        abs(float(r["hole_pred"]) - float(r["hole_gt"])) for r in rows)
    out["failure_rates"] = {
        "ssim_lt_0.50": _rate_ci(float(r["ssim"]) < 0.50 for r in rows),
        "lpips_gt_0.40": _rate_ci(float(r["lpips"]) > 0.40 for r in rows),
        "ink_iou_lt_0.20": _rate_ci(float(r["ink_iou"]) < 0.20 for r in rows),
        "frag_gt_4": _rate_ci(float(r["frag_ratio"]) > 4.0 for r in rows),
        "ink_ratio_outside_075_125": _rate_ci(
            not (0.75 <= float(r["soft_ink_ratio"]) <= 1.25) for r in rows),
    }
    writers = sorted({r["calligrapher"] for r in rows})
    for writer in writers:
        rr = [r for r in rows if r["calligrapher"] == writer]
        out["by_writer"][writer] = {
            "n": len(rr),
            "ssim": float(np.mean([float(r["ssim"]) for r in rr])),
            "lpips": float(np.mean([float(r["lpips"]) for r in rr])),
            "ink_iou": float(np.mean([float(r["ink_iou"]) for r in rr])),
            "frag": float(np.mean([float(r["frag_ratio"]) for r in rr])),
        }
    return out


def _paired(rows: list[dict[str, str]], base: list[dict[str, str]]) -> dict:
    by_id = {r["sample_id"]: r for r in base}
    common = [r for r in rows if r["sample_id"] in by_id]
    out = {"n": len(common), "metrics": {}}
    for key, higher in METRICS.items():
        delta = []
        for r in common:
            d = float(r[key]) - float(by_id[r["sample_id"]][key])
            if higher is False:
                d = -d
            elif higher is None:
                d = abs(float(by_id[r["sample_id"]][key]) - 1.0) - abs(float(r[key]) - 1.0)
            delta.append(d)
        out["metrics"][key] = _mean_ci(delta)
    return out


def cmd_summarize(args: argparse.Namespace) -> None:
    root = Path(args.root)
    arms = args.arms.split(",")
    all_rows = {arm: _rows(root / arm / "batch.csv") for arm in arms}
    report = {
        "protocol": args.protocol,
        "baseline": args.baseline,
        "seed": SEED,
        "arms": {arm: _arm_summary(rows) for arm, rows in all_rows.items()},
        "paired_vs_baseline": {},
    }
    if args.baseline:
        base = all_rows[args.baseline]
        report["paired_vs_baseline"] = {
            arm: _paired(rows, base) for arm, rows in all_rows.items() if arm != args.baseline
        }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.out_md).write_text(_summary_markdown(report), encoding="utf-8")


def _summary_markdown(report: dict) -> str:
    lines = [f"# {report['protocol']} comprehensive metrics", "",
             "Mean with paired-sample bootstrap 95% confidence intervals.", "",
             "| Arm | SSIM | LPIPS | Ink-IoU | Frag | SoftInk abs err | Severe SSIM | Ink-ratio OOR |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm, data in report["arms"].items():
        m, f = data["metrics"], data["failure_rates"]
        def fmt(x: dict, d: int = 4) -> str:
            return f"{x['mean']:.{d}f} [{x['lo']:.{d}f},{x['hi']:.{d}f}]"
        lines.append(
            f"| {arm} | {fmt(m['ssim'])} | {fmt(m['lpips'])} | {fmt(m['ink_iou'])} | "
            f"{fmt(m['frag_ratio'], 3)} | {fmt(m['soft_ink_abs_error'])} | "
            f"{100*f['ssim_lt_0.50']['mean']:.1f}% | {100*f['ink_ratio_outside_075_125']['mean']:.1f}% |")
    if report["paired_vs_baseline"]:
        lines += ["", f"## Paired improvement over {report['baseline']}", "",
                  "Positive means better after orienting lower-is-better metrics.", "",
                  "| Arm | dSSIM | dLPIPS | dInk-IoU | dFrag | dInk calibration |",
                  "|---|---:|---:|---:|---:|---:|"]
        for arm, data in report["paired_vs_baseline"].items():
            m = data["metrics"]
            def fmt(x: dict) -> str:
                return f"{x['mean']:+.4f} [{x['lo']:+.4f},{x['hi']:+.4f}]"
            lines.append(f"| {arm} | {fmt(m['ssim'])} | {fmt(m['lpips'])} | "
                         f"{fmt(m['ink_iou'])} | {fmt(m['frag_ratio'])} | {fmt(m['soft_ink_ratio'])} |")
    lines.append("")
    return "\n".join(lines)


def _skeleton_f1(a: np.ndarray, b: np.ndarray, tol: int = 2) -> float:
    from scipy.ndimage import binary_dilation
    from skimage.morphology import skeletonize
    sa, sb = skeletonize(a), skeletonize(b)
    if not sa.any() or not sb.any():
        return float("nan")
    da = binary_dilation(sa, iterations=tol)
    db = binary_dilation(sb, iterations=tol)
    precision = float((sa & db).sum() / max(sa.sum(), 1))
    recall = float((sb & da).sum() / max(sb.sum(), 1))
    return 2.0 * precision * recall / max(precision + recall, 1e-12)


def cmd_geometry(args: argparse.Namespace) -> None:
    from PIL import Image
    from src.eval.style.brush import bsd, features_from_image

    root = Path(args.samples_dir)
    ids = sorted(int(p.stem[1:]) for p in root.glob("g*.png") if (root / f"gt{p.stem[1:]}.png").exists())
    rows = []
    for idx in ids:
        gp, tp = root / f"g{idx}.png", root / f"gt{idx}.png"
        fg, ft = features_from_image(str(gp)), features_from_image(str(tp))
        d = bsd(fg, ft)
        g = np.asarray(Image.open(gp).convert("L"), np.float32) / 255.0
        t = np.asarray(Image.open(tp).convert("L"), np.float32) / 255.0
        f1s = [_skeleton_f1(g < th, t < th, 2) for th in (0.45, 0.55, 0.65)]
        wg, wt = fg.width_stats(), ft.width_stats()
        rows.append({
            "idx": idx,
            "bsd_macro": d.get("bsd_macro", float("nan")),
            "width_w1": d.get("width", float("nan")),
            "width_p50_abs": abs(wg["p50"] - wt["p50"]),
            "width_p50_ratio": wg["p50"] / max(wt["p50"], 1e-12),
            "bbox_aspect_abs": abs(fg.scalars.get("bbox_aspect", float("nan")) - ft.scalars.get("bbox_aspect", float("nan"))),
            "centroid_l2": math.hypot(
                fg.scalars.get("centroid_dx", float("nan")) - ft.scalars.get("centroid_dx", float("nan")),
                fg.scalars.get("centroid_dy", float("nan")) - ft.scalars.get("centroid_dy", float("nan"))),
            "ink_frac_abs": abs(fg.scalars.get("ink_frac", float("nan")) - ft.scalars.get("ink_frac", float("nan"))),
            "skel_f1_tol2": float(np.nanmean(f1s)),
        })
    keys = [k for k in rows[0] if k != "idx"] if rows else []
    result = {"n": len(rows), "metrics": {k: _mean_ci(r[k] for r in rows) for k in keys}, "samples": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def _font(size: int):
    from PIL import ImageFont
    candidates = [Path("C:/Windows/Fonts/msyh.ttc"), Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
                  Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")]
    for p in candidates:
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def cmd_poster(args: argparse.Namespace) -> None:
    from PIL import Image, ImageDraw

    raw = Path(args.root)
    pdir = Path(args.poster_dir)
    arms = args.arms.split(",")
    rows = {arm: _rows(raw / arm / "batch.csv") for arm in arms}
    n = min(len(x) for x in rows.values())
    # Select two high-disagreement samples per writer.  This emphasizes real
    # architectural differences while keeping every writer represented.
    scores = []
    for i in range(n):
        vals = []
        for key in ("ssim", "ink_iou", "frag_ratio", "soft_ink_ratio"):
            a = np.asarray([float(rows[arm][i][key]) for arm in arms])
            vals.append(float(a.std() / max(abs(a.mean()), 1e-6)))
        scores.append(float(np.mean(vals)))
    by_writer: dict[str, list[int]] = {}
    for i, r in enumerate(rows[arms[0]][:n]):
        by_writer.setdefault(r["calligrapher"], []).append(i)
    chosen = []
    for writer in sorted(by_writer):
        chosen.extend(sorted(by_writer[writer], key=lambda i: scores[i], reverse=True)[:args.per_writer])
    posters = {arm: Image.open(pdir / f"{arm}.png").convert("RGB") for arm in arms}
    w, _ = posters[arms[0]].size
    cell = (w - 12) // n
    gap = 6
    # Existing poster contract: input y=108, prediction y=218, GT y=328.
    y_input, y_pred, y_gt = 108, 218, 328
    out_cell, label_w, top_h = args.cell, 220, 116
    out_w = label_w + len(chosen) * out_cell
    out_h = top_h + (2 + len(arms)) * out_cell
    canvas = Image.new("RGB", (out_w, out_h), "white")
    draw = ImageDraw.Draw(canvas)
    f24, f18, f14 = _font(24), _font(18), _font(14)
    draw.text((12, 8), args.title, fill=(15, 20, 28), font=f24)
    draw.text((12, 42), "每位书家选跨模型分歧最大的样本；同列同噪声、同GT。", fill=(70, 75, 85), font=f14)
    base_rows = rows[arms[0]]
    for col, idx in enumerate(chosen):
        x = label_w + col * out_cell
        r = base_rows[idx]
        draw.text((x + 3, 65), f"{r['char']}  #{idx}", fill=(20, 20, 20), font=f18)
        draw.text((x + 3, 91), r["calligrapher"][:7], fill=(65, 65, 65), font=f14)

    def paste_row(label: str, image_arm: str, source_y: int, row_no: int, color: tuple[int, int, int]):
        y = top_h + row_no * out_cell
        draw.rectangle((0, y, label_w, y + out_cell), fill=(247, 248, 250))
        draw.text((12, y + out_cell // 2 - 14), label, fill=color, font=f18)
        im = posters[image_arm]
        for col, idx in enumerate(chosen):
            x0 = gap + idx * cell
            crop = im.crop((x0, source_y, x0 + cell, source_y + cell)).resize((out_cell, out_cell), Image.Resampling.LANCZOS)
            canvas.paste(crop, (label_w + col * out_cell, y))

    paste_row("STD", arms[0], y_input, 0, (30, 115, 170))
    paste_row("GT", arms[0], y_gt, 1, (190, 55, 55))
    for j, arm in enumerate(arms):
        m = _arm_summary(rows[arm])["metrics"]
        label = f"{arm}  S={m['ssim']['mean']:.3f} I={m['ink_iou']['mean']:.3f} F={m['frag_ratio']['mean']:.2f}"
        paste_row(label, arm, y_pred, j + 2, (25, 35, 50))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out, optimize=True)
    Path(args.out).with_suffix(".indices.json").write_text(
        json.dumps({"indices": chosen, "scores": {str(i): scores[i] for i in chosen}}, indent=2), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summarize")
    s.add_argument("--root", required=True)
    s.add_argument("--arms", required=True)
    s.add_argument("--baseline", default="")
    s.add_argument("--protocol", required=True)
    s.add_argument("--out-json", required=True)
    s.add_argument("--out-md", required=True)
    s.set_defaults(func=cmd_summarize)
    g = sub.add_parser("geometry")
    g.add_argument("--samples-dir", required=True)
    g.add_argument("--out", required=True)
    g.set_defaults(func=cmd_geometry)
    q = sub.add_parser("poster")
    q.add_argument("--root", required=True)
    q.add_argument("--poster-dir", required=True)
    q.add_argument("--arms", required=True)
    q.add_argument("--title", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--per-writer", type=int, default=2)
    q.add_argument("--cell", type=int, default=128)
    q.set_defaults(func=cmd_poster)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
