"""Seal the S5-P125 run inputs: contract + source hashes + data sizes.

serial.verify_assets() refuses to launch unless ready['contract'] equals the
plan contract, and the runner re-checks every sealed .py hash before each child
process. So this must be re-run *after* any code or CSV change, otherwise the
run halts on a stale seal (by design: it catches silent drift mid-experiment).
"""
import argparse
import hashlib
import json
from pathlib import Path

OUT = Path("assets/p125_20261030")
# Code is small, so it carries a real hash; big data is sealed by path+size only.
CODE = [
    "src/unet/style/interface.py",
    "src/unet/style/representations.py",
    "src/unet/style/injections.py",
    "src/unet/style/transfer.py",
    "src/unet/style/diagnostics.py",
    "src/unet/io.py",
    "src/unet/data.py",
    "src/train/step.py",
    "src/train/ckpt.py",
    "src/train/ema.py",
    "src/loss/flow_matching.py",
    "tools/style_matrix/runtime/train.py",
    "tools/style_matrix/runtime/common.py",
    "tools/style_matrix/runtime/serial.py",
    "tools/style_matrix/runtime/evaluate.py",
    "tools/p125/init_p1_table.py",
    "tools/p125/pilot_eg.py",
]
DATA = [
    "assets/p125_20261030/stage_p0.csv",
    "assets/p125_20261030/stage_p1.csv",
    "assets/p125_20261030/stage_p2.csv",
    "assets/p125_20261030/p0_train.csv",
    "assets/p125_20261030/char_remap_p125.json",
    "assets/p125_20261030/callig_remap_p125.json",
    "assets/p125_20261030/p125_manifest.json",
    "assets/p125_20261030/eg_refs.npz",
    "assets/u100_pretrain_20261024/validation_val100.csv",
    "assets/u100_pretrain_20261024/std_shape_bank.pt",
    "assets/u100_pretrain_20261024/tables/font_remap.json",
    "assets/u100_pretrain_20261024/tables/char_remap.json",
    "assets/u100_p1p2_20261025/p2_train.csv",
    "assets/generic48_kxl_top10_20261022/std_bbox64",
]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def size_of(path):
    p = Path(path)
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return p.stat().st_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=str(OUT / "plan_a.json"))
    ap.add_argument("--out", default=str(OUT / "ready.json"))
    a = ap.parse_args()
    plan = json.loads(Path(a.plan).read_text())
    files, total = [], 0
    for path in CODE + DATA:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError("sealed input missing: " + path)
        size = size_of(path)
        entry = dict(path=path, size=size)
        if path in CODE:
            entry["sha256"] = sha(path)
        else:
            entry["sealed"] = "path+size"
        files.append(entry)
        total += size
    for path in (OUT / "shards_all",):
        if not Path(path).exists():
            raise FileNotFoundError("latent root missing: " + str(path))
        files.append(dict(path=str(path), size=size_of(path), sealed="path+size"))
    ready = dict(contract=plan["contract"], files=files,
                 total_bytes=total, built_from=str(a.plan))
    out = Path(a.out)
    out.write_text(json.dumps(ready, ensure_ascii=False, indent=1))
    print(json.dumps(dict(out=str(out), files=len(files), total_GiB=round(total / 2 ** 30, 2),
                          contract=ready["contract"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
