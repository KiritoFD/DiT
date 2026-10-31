#!/usr/bin/env python3
"""Build a strict 20-column checkpoint comparison poster from existing images.

Manifest example: tools/milestone_poster.example.json
No model loads, no download, no inferred filename order. See Milestones.md.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str, base: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else base / p


def load_index(path: Path, mode: str = "explicit_idx") -> dict[int, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    if mode not in ("explicit_idx", "row_number"):
        raise ValueError(f"invalid sample_index_mode: {mode}")
    for ordinal, row in enumerate(rows):
        key = (row.get("idx") or row.get("index") or row.get("sample_idx")) if mode == "explicit_idx" else ordinal
        if key is None:
            raise ValueError(f"index CSV needs idx/index/sample_idx: {path}; use row_number only for historical CSV whose g{{idx}} filenames are explicitly row-numbered")
        key = int(key)
        if key in out:
            raise ValueError(f"duplicate sample idx={key}: {path}")
        out[key] = row
    return out


def font(size: int, custom: str | None) -> ImageFont.FreeTypeFont:
    candidates = [custom] if custom else []
    candidates += [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for name in candidates:
        if name and Path(name).is_file():
            return ImageFont.truetype(name, size)
    raise FileNotFoundError("No usable font found. Supply --font PATH (never copy font into archive).")


def render(manifest: dict, base: Path, output: Path, font_path: str | None, cell: int):
    ids = manifest["sample_ids"]
    if len(ids) != 20 or len(set(ids)) != 20 or any(not isinstance(x, int) for x in ids):
        raise ValueError("sample_ids must contain exactly 20 distinct integer indices")
    index_path = resolve(manifest["sample_index_csv"], base)
    index = load_index(index_path, manifest.get("sample_index_mode", "explicit_idx"))
    missing = set(ids) - set(index)
    if missing:
        raise ValueError(f"sample indices absent from authoritative CSV: {sorted(missing)}")
    required = ["character", "calligrapher", "script"]
    for idx in ids:
        for key in required:
            if not index[idx].get(key):
                raise ValueError(f"idx={idx} missing {key}")
    rows = manifest["rows"]
    if not rows:
        raise ValueError("no model rows")
    # Do not silently compare unrelated protocols, samplers or evaluation code.
    protocol = manifest["protocol_id"]
    for row in rows:
        if row.get("protocol_id") != protocol:
            raise ValueError(f"{row.get('name')} protocol mismatch: {row.get('protocol_id')} != {protocol}")
        if not row.get("method") or not row.get("name"):
            raise ValueError(f"row must contain name and method: {row}")
        if not row.get("image_pattern") or "{idx}" not in row["image_pattern"]:
            raise ValueError("image_pattern must include {idx}; never glob-sort filenames")
    label_w, top_h, caption_h, gap = 440, 110, 46, 3
    row_h = cell + 20
    width = label_w + len(ids) * (cell + gap) + 20
    height = top_h + len(rows) * row_h + caption_h
    canvas = Image.new("RGB", (width, height), "#f3f4f5")
    d = ImageDraw.Draw(canvas)
    title = font(32, font_path)
    large = font(28, font_path)
    info = font(21, font_path)
    small = font(17, font_path)
    d.text((22, 8), manifest.get("title", "Calligraphy milestones"), fill="#101015", font=title)
    d.text((22, 54), f"protocol={protocol} | same 20 sample IDs | metrics are protocol-specific",
           fill="#424753", font=info)
    for col, idx in enumerate(ids):
        x = label_w + col * (cell + gap)
        m = index[idx]
        d.text((x + 3, 73), f"{m['character']} / {idx}", fill="#101015", font=small)
    check_images = []
    for ri, row in enumerate(rows):
        y = top_h + ri * row_h
        d.rounded_rectangle((8, y, label_w - 14, y + cell), radius=9,
                            fill="#e3e7ec" if ri % 2 == 0 else "#dee5e9")
        d.text((20, y + 10), row["name"], fill="#17202c", font=large)
        d.text((20, y + 54), row["method"][:39], fill="#374457", font=info)
        met = row.get("metrics", {})
        headline = "  ".join(f"{k}: {v}" for k, v in met.items())
        # Write only source-backed metric fields. Never fill in estimated values.
        d.text((20, y + 92), headline[:48], fill="#233651", font=small)
        for col, idx in enumerate(ids):
            img = resolve(row["image_pattern"].format(idx=idx), base)
            if not img.is_file():
                raise FileNotFoundError(f"missing {row['name']} idx={idx}: {img}")
            x = label_w + col * (cell + gap)
            with Image.open(img) as im:
                im = im.convert("RGB")
                im.thumbnail((cell, cell), Image.Resampling.LANCZOS)
                tile = Image.new("RGB", (cell, cell), "white")
                tile.paste(im, ((cell - im.width) // 2, (cell - im.height) // 2))
            canvas.paste(tile, (x, y))
            check_images.append(str(img))
    footer = f"Manifest {manifest.get('manifest_id', 'unspecified')} | {len(rows)} rows × 20 chars | " \
             "WARNING: OCR/SSIM/EG only comparable with equal evaluator & protocol"
    d.text((20, height - caption_h + 10), footer, fill="#353b45", font=small)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    frozen = {
        "manifest_id": manifest.get("manifest_id"),
        "protocol_id": protocol,
        "sample_index_csv": str(index_path),
        "sample_ids": ids,
        "samples": [{k: index[idx].get(k) for k in ("character", "calligrapher", "script")}
                    | {"idx": idx} for idx in ids],
        "rows": [{"name": r["name"], "method": r["method"], "metrics": r.get("metrics", {}),
                  "image_pattern": r["image_pattern"]} for r in rows],
        "resolved_images": check_images,
        "output": str(output),
    }
    output.with_suffix(".selection.json").write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{output} ({width}x{height}); frozen={output.with_suffix('.selection.json')}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, help="JSON manifest with authoritative sample CSV")
    p.add_argument("--out", required=True)
    p.add_argument("--root", help="asset root; default repo root")
    p.add_argument("--font", help="local TTF/TTC font; never checked into Git")
    p.add_argument("--cell", type=int, default=132)
    a = p.parse_args()
    if not 96 <= a.cell <= 400:
        p.error("--cell must be in [96,400]")
    manifest = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    render(manifest, Path(a.root).resolve() if a.root else ROOT,
           Path(a.out), a.font, a.cell)


if __name__ == "__main__":
    main()
