# -*- coding: utf-8 -*-
"""Build VQ-Font data: per-font dirs -> LMDB ({font}_{uni} -> png bytes) + meta jsons.

Layout produced:
  baseline/data/vqfont/lmdb/                     keys: {font}_{UNIHEX}
  baseline/data/vqfont/meta/trainset_dict.json   {font: [uni,...]}
  baseline/data/vqfont/meta/train.json           {train:{}, avail:{}, valid:{}}
  baseline/data/vqfont/meta/all_characters.json  sorted uni list (all chars)

Fonts: 23 slots + "content_deng" (content font is kept as a trainable domain,
matching the original pipeline where the content font joins font_chosen).
"""
import io
import json
import os
import sys

import lmdb

sys.path.insert(0, os.path.dirname(__file__))
from top10_common import (DIT_ROOT, load_train_rows, renderable_chars,  # noqa: E402
                          slots_in_order)

OUT = os.path.join(DIT_ROOT, "baseline", "data", "vqfont")
CONTENT_FONT_NAME = "content_deng"


def main():
    rows = load_train_rows()
    ok = renderable_chars()
    os.makedirs(os.path.join(OUT, "meta"), exist_ok=True)

    # One image per (slot, char): the baselines' data models are font->char
    # dicts (one glyph per pair). Dedup keeps the FIRST row (img_id order),
    # matching the file that 02/04 symlink for FontDiffuser/DG-Font.
    font_unis = {}   # font -> sorted uni list
    font_char_img = {}  # (font, uni) -> source png (first occurrence)
    seen = set()
    for r in rows:
        ch = r["character"]
        if ch not in ok:
            continue
        uni = f"{ord(ch):X}"
        key = (r["slot_name"], uni)
        if key in seen:
            continue
        seen.add(key)
        font_unis.setdefault(r["slot_name"], []).append(uni)
        font_char_img[key] = os.path.join(DIT_ROOT, r["image_path"])

    content_dir = os.path.join(DIT_ROOT, "baseline", "data", "content_font", "deng")
    content_pngs = sorted(f for f in os.listdir(content_dir) if f.endswith(".png"))
    font_unis[CONTENT_FONT_NAME] = sorted(f"{ord(fn[:-4]):X}" for fn in content_pngs)
    for fn in content_pngs:
        font_char_img[(CONTENT_FONT_NAME, f"{ord(fn[:-4]):X}")] = os.path.join(content_dir, fn)
    all_unis = sorted({u for unis in font_unis.values() for u in unis})

    # ---- LMDB (fresh) ----
    # NOTE: top10 source PNGs are a RGB/L mix (~37% RGB); VQ-Font's loader
    # assumes single-channel, so normalize every entry to grayscale L.
    lmdb_path = os.path.join(OUT, "lmdb")
    import shutil as _sh
    if os.path.exists(lmdb_path):
        _sh.rmtree(lmdb_path)
    os.makedirs(lmdb_path, exist_ok=True)
    env = lmdb.open(lmdb_path, map_size=1024 ** 4)
    from PIL import Image
    import io as _io
    n = 0
    with env.begin(write=True) as txn:
        for (font, uni), png in font_char_img.items():
            im = Image.open(png).convert("L")
            buf = _io.BytesIO()
            im.save(buf, format="PNG")
            txn.put(f"{font}_{uni}".encode("utf-8"), buf.getvalue())
            n += 1
    env.close()
    print(f"lmdb wrote {n} entries (grayscale-normalized)")

    with open(os.path.join(OUT, "meta", "trainset_dict.json"), "w", encoding="utf-8") as f:
        json.dump(font_unis, f, ensure_ascii=False, indent=1)

    train = {f: unis for f, unis in font_unis.items()}
    valid = {
        "seen_fonts": [s for s in font_unis if s != CONTENT_FONT_NAME],
        "unseen_fonts": [CONTENT_FONT_NAME],   # pseudo-unseen, internal CV only
        "seen_unis": all_unis,
        "unseen_unis": all_unis,
    }
    meta = {"train": train, "avail": dict(font_unis), "valid": valid}
    with open(os.path.join(OUT, "meta", "train.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)

    with open(os.path.join(OUT, "meta", "all_characters.json"), "w", encoding="utf-8") as f:
        json.dump(all_unis, f, ensure_ascii=False)

    print(f"fonts={len(font_unis)} all_unis={len(all_unis)} "
          f"min_slot={min(len(v) for k, v in font_unis.items() if k != CONTENT_FONT_NAME)}")


if __name__ == "__main__":
    main()
