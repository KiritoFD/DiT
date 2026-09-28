# -*- coding: utf-8 -*-
"""Stage GT images for eval + verify refs.

  baseline/data/eval/strict84/gt/{slot}__{char}.png   (images from data/50k/imgs)
  baseline/data/eval/seen20/gt/{slot}__{char}.png     (images from top10 imgs)
Also emits baseline/data/eval/refs.json (deterministic 3-ref protocol).
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(__file__))
from top10_common import (DIT_ROOT, build_refs, load_eval_rows,  # noqa: E402
                          SEEN20_CSV, STRICT84_CSV)

EVAL = os.path.join(DIT_ROOT, "baseline", "data", "eval")


def main():
    build_refs(os.path.join(EVAL, "refs.json"), k=3)
    for tag, csvp in (("strict84", STRICT84_CSV), ("seen20", SEEN20_CSV)):
        gt = os.path.join(EVAL, tag, "gt")
        os.makedirs(gt, exist_ok=True)
        for r in load_eval_rows(csvp):
            dst = os.path.join(gt, f"{r['slot_name']}__{r['character']}.png")
            src = os.path.join(DIT_ROOT, r["image_path"])
            if not os.path.exists(dst):
                shutil.copyfile(src, dst)
        n = len([f for f in os.listdir(gt) if f.endswith(".png")])
        print(f"{tag}: {n} gt images -> {gt}")


if __name__ == "__main__":
    main()
