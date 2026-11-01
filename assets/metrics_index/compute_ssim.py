"""Recompute SSIM and LPIPS for every identity-verified panel against its GT.

Both metrics use the same pairing as the E-G/OCR run: panel g{i}.png vs panel
gt{i}.png, 187 rows, no resampling beyond the 256x256 panels already stored.
"""
import argparse
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
tls = __import__("threading")
torch.set_num_threads(8)

OUT = Path("assets/metrics_index/ssim_lpips_20261101.csv")


def load_rgb(p):
    with Image.open(p) as im:
        a = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def ssim_simple(a, b, c1=0.01 ** 2, c2=0.03 ** 2, win=11):
    """Gaussian-window SSIM, matches the standard skimage formulation."""
    dev, dt = a.device, a.dtype
    k = torch.arange(win, device=dev, dtype=dt) - (win - 1) / 2
    g = torch.exp(-(k ** 2) / (2 * 1.5 ** 2))
    g = (g / g.sum()).view(1, 1, 1, win)
    ga = a.mean(1, keepdim=True)
    gb = b.mean(1, keepdim=True)

    def filt(x):
        x = torch.nn.functional.conv2d(x, g, padding=(0, win // 2))
        return torch.nn.functional.conv2d(x, g.transpose(2, 3), padding=(win // 2, 0))

    mu_a, mu_b = filt(ga), filt(gb)
    sa = filt(ga * ga) - mu_a ** 2
    sb = filt(gb * gb) - mu_b ** 2
    sab = filt(ga * gb) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (sa + sb + c2))
    return float(s.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    import lpips as lpips_mod
    net = lpips_mod.LPIPS(net="alex").to(a.device).eval()
    for p in net.parameters():
        p.requires_grad_(False)

    hist = [r for r in json.loads(Path("assets/metrics_index/historical_panels.json")
                                  .read_text(encoding="utf-8")) if r.get("usable")]
    if a.limit:
        hist = hist[:a.limit]
    name_map = {}
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "mh", Path("assets/metrics_index/merge_historical.py"))
        mh = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mh)
        name_map = mh.NAME
    except Exception:  # noqa: BLE001
        pass

    rows = []
    for rec in hist:
        d = Path(rec["dir"])
        arm, step = name_map.get(rec["dir"].replace("\\", "/"), (d.name, 0))
        ss, lps = [], []
        try:
            for i in range(187):
                g, t = d / f"g{i}.png", d / f"gt{i}.png"
                if not (g.is_file() and t.is_file()):
                    break
                x, y = load_rgb(g), load_rgb(t)
                ss.append(ssim_simple(x, y))
                with torch.no_grad():
                    lps.append(float(net((x * 2 - 1).to(a.device), (y * 2 - 1).to(a.device)).item()))
            if len(ss) == 187:
                rows.append({"arm": arm, "step": step, "n": 187,
                             "ssim": round(float(np.mean(ss)), 5),
                             "lpips": round(float(np.mean(lps)), 5),
                             "source_panel": rec["dir"].replace("\\", "/")})
                print(f"{arm:14s} {step:>7} ssim={rows[-1]['ssim']:.4f} lpips={rows[-1]['lpips']:.4f}",
                      flush=True)
            else:
                print("SKIP incomplete", d, len(ss), flush=True)
        except Exception as exc:  # noqa: BLE001
            print("FAIL", d, exc, flush=True)
        if rows:
            with OUT.open("w", encoding="utf-8-sig", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
                w.writeheader()
                w.writerows(rows)
    print("done", len(rows))


if __name__ == "__main__":
    main()
