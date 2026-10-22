#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit fonts and build the raw 64x64 std-soft glyph store.

Examples
--------
Audit the full current train/eval vocabulary and build the cache::

    python tools/build_std_soft_glyphs.py \
      --csv exp-std/csv/train.csv exp-std/csv/eval200_fixed.csv \
      --out data/std_soft_glyphs_v1 --strict --preview

Build only a visual smoke set::

    python tools/build_std_soft_glyphs.py --chars "一丶你體獨㩜" \
      --out ANALYSIS/std_soft_preview --strict --preview

The PNG store contains unblurred BOX-downsampled masks.  The preview applies
``--preview-sigma`` and runtime datasets apply configurable train/eval blur.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.std_soft_glyph import (  # noqa: E402
    SCRIPT_KEYS, FontCoverage, collect_script_characters, load_font_config,
    render_base_mask, sha256_file, soften_mask,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", nargs="*", default=[])
    parser.add_argument("--chars", default="")
    parser.add_argument("--font-config", default="configs/std_soft_fonts.json")
    parser.add_argument("--font-dir", action="append", default=[])
    parser.add_argument("--out", required=True)
    parser.add_argument("--render-size", type=int, default=512)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--font-ratio", type=float, default=0.79)
    parser.add_argument("--legacy-em",action="store_true",help="Reproduce old non-bbox geometry; default uses fit256")
    parser.add_argument("--preview-sigma", type=float, default=0.65)
    parser.add_argument("--preview-max-chars", type=int, default=12,
                        help="Cap preview width; the glyph store itself is never truncated.")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def _ascii_font(size=18):
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if os.path.isfile(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def build_preview(out_dir, rows, script_specs, sigma, cell=112, max_chars=12):
    chars = []
    for row in rows:
        if row["character"] not in chars:
            chars.append(row["character"])
    chars = chars[:max(1, int(max_chars))]
    label_w, header_h = 230, 34
    canvas = Image.new("RGB", (label_w + cell * len(chars), header_h + cell * 3), "white")
    draw = ImageDraw.Draw(canvas)
    font = _ascii_font(17)
    draw.text((8, 7), f"soft preview sigma={sigma:g}", fill="black", font=font)
    for col, char in enumerate(chars):
        draw.text((label_w + col * cell + 8, 7), f"U+{ord(char):04X}", fill="black", font=font)
    for row_i, sid in enumerate((0, 3, 4)):
        y = header_h + row_i * cell
        chosen = [r for r in rows if int(r["script_id"]) == sid]
        primary = script_specs[sid]["faces"][0].label if script_specs[sid]["faces"] else "UNRESOLVED"
        draw.text((8, y + 8), f"{SCRIPT_KEYS[sid]}: {primary}", fill="black", font=font)
        by_char = {r["character"]: r for r in chosen}
        for col, char in enumerate(chars):
            record = by_char.get(char)
            if record is None:
                continue
            image = Image.open(os.path.join(out_dir, record["path"])).convert("L")
            image = soften_mask(image, sigma).resize((cell - 12, cell - 12), Image.Resampling.NEAREST)
            tile = Image.new("L", (cell, cell), 255)
            tile.paste(Image.eval(image, lambda p: 255 - p), (6, 6))
            canvas.paste(tile.convert("RGB"), (label_w + col * cell, y))
            if int(record["font_rank"]) > 0:
                short_name = os.path.splitext(record['font_file'])[0].replace(
                    "NotoSerifSC-VF", "NotoSerifSC")
                draw.text((label_w + col * cell + 4, y + cell - 19),
                          short_name, fill=(180, 0, 0), font=_ascii_font(9))
    path = os.path.join(out_dir, "preview.png")
    canvas.save(path)
    return path


def main():
    args = parse_args()
    if not args.csv and not args.chars:
        raise SystemExit("provide --csv and/or --chars")
    config, scripts, font_dirs = load_font_config(args.font_config, args.font_dir)
    chars_by_script = collect_script_characters(args.csv, args.chars)
    coverage = FontCoverage()

    print("[fonts] resolved cascades")
    for sid in (0, 3, 4):
        spec = scripts[sid]
        print(f"  {sid}/{spec['key']}: {[f.label for f in spec['faces']]}"
              f" unresolved={spec['unresolved']}")
        if not spec["faces"]:
            raise SystemExit(f"no font resolved for script {sid}; searched {font_dirs}")
        for rank, face in enumerate(spec["faces"]):
            count = sum(coverage.supports(face, ch) for ch in chars_by_script[sid])
            print(f"    [{rank}] {face.label}: {count}/{len(chars_by_script[sid])}")

    selected = {}
    missing = []
    selected_counts = {sid: Counter() for sid in SCRIPT_KEYS}
    for sid in (0, 3, 4):
        faces = scripts[sid]["faces"]
        for char in sorted(chars_by_script[sid], key=ord):
            face = coverage.choose(faces, char)
            if face is None:
                missing.append((sid, char))
            else:
                rank = faces.index(face)
                selected[(sid, char)] = (face, rank)
                selected_counts[sid][face.label] += 1
    for sid in (0, 3, 4):
        print(f"[select] {sid}/{SCRIPT_KEYS[sid]}: {dict(selected_counts[sid])}")
    print(f"[coverage] selected={len(selected)} missing={len(missing)}")
    if args.audit_only:
        if missing and args.strict:
            raise SystemExit(f"missing glyphs: {missing[:20]}")
        return

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    records = []
    render_failures = []
    for (sid, char), (face, rank) in selected.items():
        key = scripts[sid]["key"]
        rel = os.path.join(key, f"U+{ord(char):05X}.png").replace("\\", "/")
        path = os.path.join(out_dir, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            image = render_base_mask(
                char, face, render_size=args.render_size,
                output_size=args.size, font_ratio=args.font_ratio,bbox_normalize=not args.legacy_em)
            image.save(path)
        except Exception as exc:  # keep all failures visible in missing.txt
            render_failures.append((sid, char, f"{type(exc).__name__}: {exc}"))
            continue
        records.append({
            "script_id": sid,
            "script_key": key,
            "character": char,
            "codepoint": f"U+{ord(char):05X}",
            "path": rel,
            "font_file": os.path.basename(face.path),
            "font_path": face.path,
            "font_face": face.face,
            "font_rank": rank,
        })
    with open(os.path.join(out_dir, "index.csv"), "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()) if records else [
            "script_id", "script_key", "character", "codepoint", "path",
            "font_file", "font_path", "font_face", "font_rank"])
        writer.writeheader()
        writer.writerows(records)
    with open(os.path.join(out_dir, "missing.txt"), "w", encoding="utf-8") as handle:
        for sid, char in missing:
            handle.write(f"{sid}\t{SCRIPT_KEYS[sid]}\tU+{ord(char):05X}\t{char}\tno cmap coverage\n")
        for sid, char, reason in render_failures:
            handle.write(f"{sid}\t{SCRIPT_KEYS[sid]}\tU+{ord(char):05X}\t{char}\t{reason}\n")
    used_faces = sorted({face for face, _rank in selected.values()}, key=lambda f: f.label)
    provenance = {
        "version": 1,
        "font_config": os.path.abspath(args.font_config),
        "font_config_sha256": sha256_file(args.font_config),
        "input_csvs": [{"path": os.path.abspath(p), "sha256": sha256_file(p)} for p in args.csv],
        "render_size": args.render_size,
        "output_size": args.size,
        "font_ratio": args.font_ratio,
        "geometry": "legacy_fixed_em" if args.legacy_em else "true_ink_bbox_fit256_aspect_preserved",
        "stored_blur_sigma": 0.0,
        "preview_sigma": args.preview_sigma,
        "polarity": "ink=255,background=0",
        "fonts": [{"path": f.path, "face": f.face, "sha256": sha256_file(f.path)}
                  for f in used_faces],
        "selection_counts": {str(sid): dict(counts) for sid, counts in selected_counts.items()},
        "missing_count": len(missing) + len(render_failures),
    }
    with open(os.path.join(out_dir, "provenance.json"), "w", encoding="utf-8") as handle:
        json.dump(provenance, handle, ensure_ascii=False, indent=2)
    if args.preview and records:
        print(f"[preview] {build_preview(out_dir, records, scripts, args.preview_sigma, max_chars=args.preview_max_chars)}")
    print(f"[done] {len(records)} glyphs -> {out_dir}")
    if (missing or render_failures) and args.strict:
        raise SystemExit(f"strict build failed: cmap_missing={len(missing)} "
                         f"render_failures={len(render_failures)}; see missing.txt")


if __name__ == "__main__":
    main()
