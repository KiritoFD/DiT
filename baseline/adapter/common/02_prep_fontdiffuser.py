# -*- coding: utf-8 -*-
"""Build FontDiffuser data layout from top10_style23 (full 38,583 rows).

  baseline/data/fontdiffuser/train/ContentImage/{char}.png
  baseline/data/fontdiffuser/train/TargetImage/{slot_name}/{slot_name}+{char}.png

Then patch dataset/font_dataset.py (.jpg -> .png) so the loader reads our PNGs.
Rows whose char has no content-font rendering are skipped (recorded).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from top10_common import (DIT_ROOT, load_train_rows, content_png,  # noqa: E402
                          renderable_chars, slots_in_order)

FD_ROOT = os.path.join(DIT_ROOT, "baseline", "data", "fontdiffuser", "train")
FD_REPO = os.path.join(DIT_ROOT, "baseline", "FontDiffuser")


def link(src, dst):
    if os.path.exists(dst):
        return
    try:
        os.symlink(os.path.abspath(src), dst)
    except OSError:
        os.link(os.path.abspath(src), dst)


def main():
    rows = load_train_rows()
    ok = renderable_chars()
    _, slot_files = slots_in_order(rows)
    skipped = 0
    for slot, files in slot_files.items():
        os.makedirs(os.path.join(FD_ROOT, "TargetImage", slot), exist_ok=True)
    os.makedirs(os.path.join(FD_ROOT, "ContentImage"), exist_ok=True)

    n_tgt = 0
    for r in rows:
        ch = r["character"]
        if ch not in ok:
            skipped += 1
            continue
        tgt_name = f"{r['slot_name']}+{ch}.png"
        link(os.path.join(DIT_ROOT, r["image_path"]),
             os.path.join(FD_ROOT, "TargetImage", r["slot_name"], tgt_name))
        link(content_png(ch), os.path.join(FD_ROOT, "ContentImage", f"{ch}.png"))
        n_tgt += 1

    # patch loader extensions (.jpg -> .png), idempotent
    fd_path = os.path.join(FD_REPO, "dataset", "font_dataset.py")
    src = open(fd_path, encoding="utf-8").read()
    if ".jpg" in src:
        open(fd_path, "w", encoding="utf-8").write(src.replace(".jpg", ".png"))
        print("patched font_dataset.py: .jpg -> .png")

    print(f"linked {n_tgt} target pairs (skipped {skipped} rows w/o content render)")
    print(f"slots: {len(slot_files)}, content chars: {len(ok)}")


if __name__ == "__main__":
    main()
