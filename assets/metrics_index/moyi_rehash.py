"""Recover MoYi GT identity by exact/near hash against every known mosaic source.

Strategy (in order of strictness):
  1. exact byte hash of the raw file
  2. exact hash of the decoded RGB array
  3. exact hash of the decoded grayscale array
  4. exact hash after inverting polarity
  5. near-hash: identical after rounding to 8 grey levels (compression tolerance)
Each hit must be unique across the candidate pool, otherwise it is rejected.
"""
import csv
import hashlib
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
PANEL = Path("assets/server_48_evals/moyi_12ch_eval200fix")
OUT = Path("assets/metrics_index/moyi_hash_identity.json")

CSV_CANDIDATES = [
    "exp-std/csv/eval200_fixed.csv",
    "exp-std/csv/eval200.csv",
    "exp-std/csv/eval200_combo_unseen.csv",
    "assets/u100_pretrain_20261024/validation_val100.csv",
    "assets/data_clean/top10_20261021_v1/scan/inputs/train.csv",
    "assets/data_clean/top10_20261021_v1/candidate_reviewed/train_top10_candidate.csv",
    "_local_ckpt/from48/all/exp-std/csv/train_top10_noaug.csv",
]


def hashes(path):
    out = {}
    try:
        raw = Path(path).read_bytes()
    except OSError:
        return out
    out["file"] = hashlib.sha256(raw).hexdigest()
    with Image.open(path) as im:
        a = np.asarray(im.convert("RGB"))
        out["rgb"] = hashlib.sha256(a.tobytes()).hexdigest()
        g = np.asarray(im.convert("L"))
        out["l"] = hashlib.sha256(g.tobytes()).hexdigest()
        out["l_inv"] = hashlib.sha256((255 - g).tobytes()).hexdigest()
        q = (g // 32) * 32  # 8-level quantisation tolerates JPEG-ish noise
        out["l_q8"] = hashlib.sha256(q.tobytes()).hexdigest()
    return out


def build_index():
    """hash -> list of (csv, row_index) across every candidate protocol file."""
    index = defaultdict(list)
    total = 0
    for c in CSV_CANDIDATES:
        p = Path(c)
        if not p.is_file():
            print("  (missing csv)", c)
            continue
        rows = list(csv.DictReader(open(p, encoding="utf-8-sig")))
        usable = [r for r in rows if r.get("image_path") and Path(r["image_path"]).is_file()]
        print(f"  indexing {c}: {len(usable)}/{len(rows)} rows with readable images")
        for i, r in enumerate(usable):
            for k, h in hashes(r["image_path"]).items():
                index[(k, h)].append((c, i, r))
            total += 1
    print("indexed images", total)
    return index


def main():
    print("building hash index...")
    index = build_index()

    recovered = []
    stats = defaultdict(int)
    for i in range(187):
        gp = PANEL / f"gt{i}.png"
        if not gp.is_file():
            break
        h = hashes(gp)
        hit = None
        mode = None
        for key in ("file", "rgb", "l", "l_inv", "l_q8"):
            cands = index.get((key, h.get(key)), [])
            if len(cands) == 1:
                hit, mode = cands[0], key
                break
            if len(cands) > 1:
                stats["ambiguous"] += 1
                break
        if hit:
            stats[mode] += 1
            row = hit[2]
            recovered.append({"gt_index": i, "mode": mode, "csv": hit[0],
                              "row": hit[1], "img_id": row.get("img_id", ""),
                              "character": row.get("character", ""),
                              "script": row.get("script", ""),
                              "calligrapher": row.get("calligrapher", "")})
        else:
            stats["miss"] += 1

    print("stats", dict(stats))
    print("recovered", len(recovered), "/ 187")
    if recovered:
        with OUT.open("w", encoding="utf-8") as f:
            json.dump(recovered, f, ensure_ascii=False, indent=1)
        for r in recovered[:8]:
            print(" ", r["gt_index"], r["mode"], r["character"], r["calligrapher"])
    else:
        print("NOT recovered")


if __name__ == "__main__":
    main()
