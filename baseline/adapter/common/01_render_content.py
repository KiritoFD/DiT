# -*- coding: utf-8 -*-
"""Render Deng.ttf content font -> 256x256 white-bg black-ink PNGs, one per char.

Output: baseline/data/content_font/deng/{char}.png  (grayscale L)
Also writes charset manifests. Chars the font cannot render are recorded and
skipped everywhere downstream (they are Ext-B/I and absent from eval sets).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from top10_common import DIT_ROOT, load_train_rows  # noqa: E402

from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from fontTools.ttLib import TTFont  # noqa: E402

FONT_PATH = os.path.join(DIT_ROOT, "_fonts", "Deng.ttf")
OUT_DIR = os.path.join(DIT_ROOT, "baseline", "data", "content_font", "deng")
IMG = 256
GLYPH_BOX = 224  # glyph fitted into GLYPH_BOX x GLYPH_BOX, centered


def glyph_bbox(font, char):
    from PIL import features
    dummy = Image.new("L", (IMG * 2, IMG * 2), 255)
    d = ImageDraw.Draw(dummy)
    d.text((IMG // 2, IMG // 2), char, font=font, fill=0, anchor="mm")
    return dummy.getbbox()


def render(font, char):
    bbox = glyph_bbox(font, char)
    if bbox is None or (bbox[2] - bbox[0]) <= 1 or (bbox[3] - bbox[1]) <= 1:
        return None
    pad = 8
    tile = Image.new("L", (IMG * 2, IMG * 2), 255)
    d = ImageDraw.Draw(tile)
    d.text((IMG, IMG), char, font=font, fill=0, anchor="mm")
    bb = tile.getbbox()
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    scale = min((GLYPH_BOX - 2 * pad) / w, (GLYPH_BOX - 2 * pad) / h, 1.0)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    crop = tile.crop(bb).resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("L", (IMG, IMG), 255)
    canvas.paste(crop, ((IMG - nw) // 2, (IMG - nh) // 2))
    return canvas


def main():
    rows = load_train_rows()
    chars = sorted({r["character"] for r in rows})
    tt = TTFont(FONT_PATH)
    cmap = set()
    for t in tt["cmap"].tables:
        cmap |= set(t.cmap.keys())
    have = [c for c in chars if ord(c) in cmap]
    miss = [c for c in chars if ord(c) not in cmap]
    print(f"chars={len(chars)} renderable={len(have)} missing={len(miss)}")

    os.makedirs(OUT_DIR, exist_ok=True)
    font = ImageFont.truetype(FONT_PATH, IMG * 2)
    n_ok = 0
    for i, c in enumerate(have):
        out = os.path.join(OUT_DIR, f"{c}.png")
        if os.path.exists(out):
            n_ok += 1
            continue
        im = render(font, c)
        if im is None:
            miss.append(c)
            continue
        im.save(out)
        n_ok += 1
        if (i + 1) % 1000 == 0:
            print(f"  rendered {i + 1}/{len(have)}")

    man = {"img": IMG, "font": FONT_PATH,
           "renderable": sorted(n for n in os.listdir(OUT_DIR) if n.endswith(".png")),
           "missing": sorted(set(miss))}
    with open(os.path.join(OUT_DIR, "charset_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=1)
    print(f"done: {len(man['renderable'])} pngs, {len(man['missing'])} missing -> {man['missing']}")


if __name__ == "__main__":
    main()
