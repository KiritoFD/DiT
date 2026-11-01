"""Collect SSIM / LPIPS for every identity-verified panel.

Sources, in priority order per panel:
  1. sibling metrics json inside the panel dir (metrics.json / eval_auto.json)
  2. run-level checkpoints/eval_auto_<step>.json
  3. run-level eval_metrics.json under an eval_step dir
Panels with no recorded value are simply absent (never imputed).
"""
import csv
import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
OUT = Path("assets/metrics_index/ssim_lpips_20261101.csv")

# arm/step labels reuse the curated map from merge_historical when available.
NAME = {}
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "mh", Path("assets/metrics_index/merge_historical.py"))
    mh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mh)
    NAME = mh.NAME
except Exception as exc:  # noqa: BLE001
    print("NAME map unavailable:", exc)

KEYS_SSIM = ("ssim", "ssim_mean", "strict_ssim", "eval_ssim")
KEYS_LP = ("lpips", "lpips_mean", "strict_lpips", "eval_lpips")


def pick(d, keys):
    for k in keys:
        if k in d and isinstance(d[k], (int, float)):
            return float(d[k])
    return None


def from_json(p):
    try:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return None, None
    if isinstance(d, dict):
        for sub in ("metrics", "content", "result", "eval200fix"):
            if isinstance(d.get(sub), dict):
                d2 = {**d, **d[sub]}
                s, l = pick(d2, KEYS_SSIM), pick(d2, KEYS_LP)
                if s is not None or l is not None:
                    return s, l
        return pick(d, KEYS_SSIM), pick(d, KEYS_LP)
    return None, None


def load_hist():
    p = Path("assets/metrics_index/historical_panels.json")
    return [r for r in json.loads(p.read_text(encoding="utf-8")) if r.get("usable")]


def main():
    rows = []
    for rec in load_hist():
        d = Path(rec["dir"])
        arm, step = NAME.get(rec["dir"].replace("\\", "/"), (d.name, 0))
        s = l = None
        src = ""
        # 1. inside the panel dir
        for cand in sorted(d.glob("*.json")):
            s2, l2 = from_json(cand)
            if s2 is not None or l2 is not None:
                s, l, src = s2, l2, str(cand)
                break
        # 2. run-level checkpoints/eval_auto_<step>.json
        if s is None:
            for p in [d, *d.parents]:
                if (p / "checkpoints").is_dir():
                    for f in sorted((p / "checkpoints").glob("eval_auto_*.json")):
                        digits = f.stem.replace("eval_auto_", "")
                        if digits.isdigit() and int(digits) == step:
                            s2, l2 = from_json(f)
                            if s2 is not None or l2 is not None:
                                s, l, src = s2, l2, str(f)
                    break
        # 3. sibling eval_metrics.json
        if s is None:
            for p in [d, *d.parents]:
                if (p / "eval_metrics.json").is_file():
                    s2, l2 = from_json(p / "eval_metrics.json")
                    s, l, src = s2, l2, str(p / "eval_metrics.json")
                    break
        rows.append({"arm": arm, "step": step, "n": 187,
                     "ssim": round(s, 5) if s is not None else "",
                     "lpips": round(l, 5) if l is not None else "",
                     "source_panel": rec["dir"].replace("\\", "/"),
                     "source_metric": src.replace("\\", "/")})
    with OUT.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    got = sum(1 for r in rows if r["ssim"] != "" and r["lpips"] != "")
    print(f"panels {len(rows)}, with both metrics {got}")
    for r in rows:
        if r["ssim"] != "":
            print(f"  {r['arm']:14s} {r['step']:>7} ssim={r['ssim']} lpips={r['lpips']}")


if __name__ == "__main__":
    main()
