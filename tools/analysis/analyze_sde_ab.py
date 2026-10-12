#!/usr/bin/env python3
"""Paired diagnosis for deterministic ODE vs corrected-SDE evaluation outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


METRICS = (
    "ssim", "mse", "lpips", "ink_ssim", "ink_iou", "skel_iou",
    "frag_ratio", "hole_pred",
)


def read_rows(path: Path) -> dict[int, dict[str, float]]:
    rows = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            idx = int(row["idx"])
            rows[idx] = {
                key: float(row[key]) for key in METRICS if row.get(key, "") != ""
            }
    return rows


def image_stats(path: Path) -> dict[str, float]:
    x = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    gx = np.zeros_like(x)
    gy = np.zeros_like(x)
    gx[:, 1:-1] = x[:, 2:] - x[:, :-2]
    gy[1:-1, :] = x[2:, :] - x[:-2, :]
    lap = np.zeros_like(x)
    lap[1:-1, 1:-1] = (
        4 * x[1:-1, 1:-1] - x[:-2, 1:-1] - x[2:, 1:-1]
        - x[1:-1, :-2] - x[1:-1, 2:]
    )
    fft = np.fft.fftshift(np.abs(np.fft.fft2(x - x.mean())) ** 2)
    h, w = x.shape
    yy, xx = np.mgrid[0:h, 0:w]
    radius = np.sqrt(((yy - h / 2) / (h / 2)) ** 2
                     + ((xx - w / 2) / (w / 2)) ** 2)
    return {
        "ink": float((x < 0.5).mean()),
        "grad": float(np.sqrt(gx ** 2 + gy ** 2).mean()),
        "lapvar": float(lap.var()),
        "hf": float(fft[radius > 0.5].sum() / max(fft.sum(), 1e-12)),
    }


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator,
                 draws: int = 20000) -> list[float]:
    n = len(values)
    means = np.empty(draws, dtype=np.float64)
    chunk = 1000
    for start in range(0, draws, chunk):
        size = min(chunk, draws - start)
        take = rng.integers(0, n, size=(size, n))
        means[start:start + size] = values[take].mean(axis=1)
    return [float(v) for v in np.percentile(means, [2.5, 97.5])]


def delta_summary(delta: np.ndarray, rng: np.random.Generator) -> dict[str, object]:
    return {
        "mean": float(delta.mean()),
        "median": float(np.median(delta)),
        "p10": float(np.percentile(delta, 10)),
        "p90": float(np.percentile(delta, 90)),
        "ci95_mean": bootstrap_ci(delta, rng),
        "positive": int((delta > 0).sum()),
        "negative": int((delta < 0).sum()),
        "zero": int((delta == 0).sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ode-csv", type=Path, required=True)
    ap.add_argument("--sde-csv", type=Path, required=True)
    ap.add_argument("--ode-images", type=Path, required=True)
    ap.add_argument("--sde-images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ode = read_rows(args.ode_csv)
    sde = read_rows(args.sde_csv)
    ids = sorted(set(ode) & set(sde))
    if not ids:
        raise RuntimeError("No paired rows")
    rng = np.random.default_rng(20261012)

    report: dict[str, object] = {"n": len(ids), "metrics": {}}
    metric_deltas = {}
    for key in METRICS:
        ov = np.array([ode[i][key] for i in ids], dtype=np.float64)
        sv = np.array([sde[i][key] for i in ids], dtype=np.float64)
        delta = sv - ov
        metric_deltas[key] = delta
        report["metrics"][key] = {
            "ode_mean": float(ov.mean()),
            "sde_mean": float(sv.mean()),
            "sde_minus_ode": delta_summary(delta, rng),
        }

    image_rows = {"ode": [], "sde": [], "gt": [], "mae_ode_sde": []}
    for i in ids:
        op = args.ode_images / f"g{i}.png"
        sp = args.sde_images / f"g{i}.png"
        gp = args.ode_images / f"gt{i}.png"
        image_rows["ode"].append(image_stats(op))
        image_rows["sde"].append(image_stats(sp))
        image_rows["gt"].append(image_stats(gp))
        oi = np.asarray(Image.open(op).convert("L"), dtype=np.float32) / 255.0
        si = np.asarray(Image.open(sp).convert("L"), dtype=np.float32) / 255.0
        image_rows["mae_ode_sde"].append(float(np.abs(oi - si).mean()))

    report["image_stats"] = {}
    for key in ("ink", "grad", "lapvar", "hf"):
        ov = np.array([x[key] for x in image_rows["ode"]])
        sv = np.array([x[key] for x in image_rows["sde"]])
        gv = np.array([x[key] for x in image_rows["gt"]])
        report["image_stats"][key] = {
            "gt_mean": float(gv.mean()),
            "ode_mean": float(ov.mean()),
            "sde_mean": float(sv.mean()),
            "ode_over_gt": float(ov.mean() / max(gv.mean(), 1e-12)),
            "sde_over_gt": float(sv.mean() / max(gv.mean(), 1e-12)),
            "sde_minus_ode": delta_summary(sv - ov, rng),
        }
    report["image_stats"]["mae_ode_sde"] = {
        "mean": float(np.mean(image_rows["mae_ode_sde"])),
        "median": float(np.median(image_rows["mae_ode_sde"])),
    }

    d_ssim = metric_deltas["ssim"]
    frag_improvement = -metric_deltas["frag_ratio"]
    d_hf = np.array([x["hf"] for x in image_rows["sde"]]) - np.array(
        [x["hf"] for x in image_rows["ode"]])
    report["relationships"] = {
        "corr_dssim_frag_improvement": float(np.corrcoef(d_ssim, frag_improvement)[0, 1]),
        "corr_dssim_dhf": float(np.corrcoef(d_ssim, d_hf)[0, 1]),
    }

    base_frag = np.array([ode[i]["frag_ratio"] for i in ids])
    q25, q75 = np.percentile(base_frag, [25, 75])
    low = base_frag <= q25
    high = base_frag >= q75
    report["frag_strata"] = {
        "bottom_quartile": {
            "n": int(low.sum()),
            "ode_frag_mean": float(base_frag[low].mean()),
            "dssim_mean": float(d_ssim[low].mean()),
            "frag_improvement_mean": float(frag_improvement[low].mean()),
        },
        "top_quartile": {
            "n": int(high.sum()),
            "ode_frag_mean": float(base_frag[high].mean()),
            "dssim_mean": float(d_ssim[high].mean()),
            "frag_improvement_mean": float(frag_improvement[high].mean()),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
