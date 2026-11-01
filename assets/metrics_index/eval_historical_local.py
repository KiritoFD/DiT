"""Run pooled E-G + OCR on identity-verified local historical panels (187-row protocol).

Uses the same evaluator code as the remote run; only the input panels differ.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from tools.eval_unpaired.domain.statistics import verification
from tools.eval_unpaired.protocol import reference_bank
from tools.eval_unpaired.networks import Evaluators

ROOT = Path(".")
CSV = ROOT / "exp-std/csv/eval200_fixed.csv"
OUT = ROOT / "assets/metrics_index/historical_eval.json"


def panel_list():
    inv = json.loads((ROOT / "assets/metrics_index/historical_panels.json").read_text())
    return [r for r in inv if r.get("usable")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    panels = panel_list()
    if a.limit:
        panels = panels[:a.limit]
    print("panels", len(panels), flush=True)

    rows = list(csv.DictReader(CSV.open(encoding="utf-8-sig")))
    # Use the exact same reference bank as the remote 121-panel run, so pooled AUCs
    # are directly comparable (a 187-row eval CSV cannot supply K64 refs by itself).
    ref_rows = ROOT / "assets/metrics_index/ref_rows.json"
    ref_feat = ROOT / "assets/metrics_index/ref_features.npz"
    refs = json.loads(ref_rows.read_text(encoding="utf-8"))
    print("refs", len(refs), "panel rows", len(rows), flush=True)
    nets = Evaluators(a.device, False, None, [])
    bank = dict(np.load(ref_feat))
    print("bank keys", list(bank), "n", len(next(iter(bank.values()))), flush=True)

    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    for rec in panels:
        d = Path(rec["dir"])
        if rec["dir"] in results and "error" not in results[rec["dir"]]:
            continue
        try:
            gen = [str(d / f"g{i}.png") for i in range(len(rows))]
            feats = nets.extract(gen, 32)
            item = {}
            for name, x in feats.items():
                if name.startswith("E-"):
                    v = verification(x, bank[name], rows, refs)
                    item[name] = {k: val for k, val in v.items() if k != "rows"}
            from tools.f_20261101.eval_arm import ocr_and_ink
            ocr, ink, n = ocr_and_ink(d, rows, a.device)
            item["ocr"] = ocr
            item["ink"] = ink
            item["n_png"] = n
            results[rec["dir"]] = item
            pooled = item.get("E-G", {}).get("pooled", {})
            print(rec["dir"], "n", n, "auc", round(100 * pooled.get("auc", 0), 2),
                  "ocr", ocr.get("top1"), flush=True)
        except Exception as exc:  # noqa: BLE001
            results[rec["dir"]] = {"error": f"{type(exc).__name__}: {exc}"}
            print("FAIL", rec["dir"], exc, flush=True)
        OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print("done", len(results))


if __name__ == "__main__":
    main()
