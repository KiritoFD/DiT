# -*- coding: utf-8 -*-
"""把 eval_samples 里的 sample/gt 成对拼成一张对比网格图，便于目检。

用法（远程）:
  python _make_vis_grid.py --dir <eval_samples/stepXXXXX> --n 8 --out /tmp/vis.png
"""
import os, sys, argparse, random
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="/tmp/vis.png")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cell", type=int, default=256)
    args = ap.parse_args()

    fs = sorted(f for f in os.listdir(args.dir) if f.startswith("sample"))
    random.seed(args.seed)
    pick = random.sample(fs, min(args.n, len(fs)))
    print("picked:", pick)

    cell = args.cell
    pad = 8
    cols = 4                      # sample,gt, sample,gt
    rows = (len(pick) + 1) // 2
    W = cols * cell + (cols + 1) * pad
    H = rows * cell + (rows + 1) * pad
    canvas = Image.new("RGB", (W, H), (255, 255, 255))

    def paste(i, j, path):
        try:
            im = Image.open(path).convert("RGB").resize((cell, cell))
        except Exception as e:
            print(f"  skip {path}: {e}")
            return
        x = pad + j * (cell + pad)
        y = pad + i * (cell + pad)
        canvas.paste(im, (x, y))

    for k, name in enumerate(pick):
        idx = name[len("sample"):]           # e.g. "105.png"
        r, c = divmod(k, 2)
        paste(r, c * 2, os.path.join(args.dir, f"sample{idx}"))
        paste(r, c * 2 + 1, os.path.join(args.dir, f"gt{idx}"))

    canvas.save(args.out)
    print(f"saved {args.out}  ({W}x{H})  layout: {rows} rows x 2 pairs "
          f"(left=sample, right=gt)")


if __name__ == "__main__":
    main()
