# -*- coding: utf-8 -*-
"""Build content-font latent shards using the PROJECT's own vae_io infra.

Content PNGs (Deng.ttf renders, one per char) -> sd-vae-ft-ema latents,
stored as shard_*.npz with img_id = ord(char) in
  baseline/data/content_font/deng_shards/

Also verifies the target shards (data/top10_style23/shards_img) are present.
"""
import csv
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
from top10_common import DIT_ROOT, CONTENT_FONT_DIR, renderable_chars  # noqa: E402

OUT_CSV = f"{DIT_ROOT}/baseline/data/content_font/deng_latent.csv"
OUT_SHARDS = f"{DIT_ROOT}/baseline/data/content_font/deng_shards"


def main():
    ok = renderable_chars()
    # their vae_io._parse_rows derives img_id from `<digits>.png` filenames,
    # so expose the renders under numeric (codepoint) names via symlinks.
    num_dir = f"{CONTENT_FONT_DIR}_num"
    os.makedirs(num_dir, exist_ok=True)
    for ch in ok:
        dst = f"{num_dir}/{ord(ch)}.png"
        if not os.path.exists(dst):
            try:
                os.symlink(f"{CONTENT_FONT_DIR}/{ch}.png", dst)
            except OSError:
                os.link(f"{CONTENT_FONT_DIR}/{ch}.png", dst)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "img_id"])
        for ch in sorted(ok, key=ord):
            w.writerow([f"{num_dir}/{ord(ch)}.png", ord(ch)])
    print(f"content csv: {len(ok)} chars -> {OUT_CSV}")

    if not os.path.exists(f"{OUT_SHARDS}/shard_00000.npz"):
        r = subprocess.run(
            ["/opt/conda/envs/baseline/bin/python", "-m", "src.data.vae_io",
             "--csv", OUT_CSV, "--out", OUT_SHARDS, "--transform", "gray",
             "--batch", "64"],
            cwd=DIT_ROOT, capture_output=True, text=True)
        print(r.stdout[-800:])
        print(r.stderr[-800:])
        if r.returncode != 0:
            raise SystemExit("vae_io failed")
    else:
        print("content shards exist, skip")

    tgt = f"{DIT_ROOT}/data/top10_style23/shards_img"
    n = len([f for f in os.listdir(tgt) if f.endswith(".npz")])
    print(f"target shards OK: {n} files in {tgt}")


if __name__ == "__main__":
    main()
