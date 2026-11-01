"""Evaluate MoYi panels with the recovered protocol (characters + writers)."""
import argparse
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(".")
PROTO = ROOT / "assets/metrics_index/moyi12ch_protocol.csv"
OUT = ROOT / "assets/metrics_index/moyi_eval.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    from tools.eval_unpaired.domain.statistics import verification
    from tools.eval_unpaired.networks import Evaluators
    from tools.f_20261101.eval_arm import ocr_and_ink

    proto = list(csv.DictReader(PROTO.open(encoding="utf-8-sig")))
    rows = [{"character": r["character"], "script": r["script"],
             "calligrapher": r["calligrapher"], "img_id": r["img_id"]} for r in proto]
    refs = json.loads((ROOT / "assets/metrics_index/ref_rows.json").read_text(encoding="utf-8"))
    bank = dict(np.load(ROOT / "assets/metrics_index/ref_features.npz"))
    print("refs", len(refs), "rows", len(rows))

    nets = Evaluators(a.device, False, None, [])
    panels = {
        "MoYi-12ch@50k": ROOT / "assets/server_48_evals/moyi_12ch_eval200fix",
        "MoYi-4ch@80k": ROOT / "assets/style_eval_src/moyi_top10_rf_4ch",
    }
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name, d in panels.items():
        if not d.is_dir():
            print("MISSING", name, d)
            continue
        gen = [str(d / f"g{i}.png") for i in range(187)]
        feats = nets.extract(gen, 32)
        item = {}
        for k, x in feats.items():
            if k.startswith("E-"):
                v = verification(x, bank[k], rows, refs)
                item[k] = {kk: vv for kk, vv in v.items() if kk != "rows"}
        ocr, ink, n = ocr_and_ink(d, rows, a.device)
        item["ocr"], item["ink"], item["n_png"] = ocr, ink, n
        res[name] = item
        p = item.get("E-G", {}).get("pooled", {})
        print(f"{name}: E-G {100 * p.get('auc', 0):.2f} (n_q={item.get('E-G', {}).get('n_queries')}) "
              f"OCR {ocr.get('top1')}", flush=True)
        OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("done")


if __name__ == "__main__":
    main()
