"""Re-render the 256px canonical rasters behind an existing bbox STD store.

The store keeps only the 64px conditions; the strong shape-augmentation bank
needs the true 256px raster per (script_id, character). Rendering is
deterministic from the frozen font assignments in the store index, and every
render must BOX-downsample to the stored 64px image bit-exactly; any mismatch
fails the build instead of silently changing training geometry.
"""
import argparse
import csv
import hashlib
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.std_soft_glyph import FontFace, render_base_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--store', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    store = Path(args.store); out = Path(args.out)
    if out.exists(): raise FileExistsError(out)
    rows = list(csv.DictReader((store / 'index.csv').open(encoding='utf-8-sig')))
    keys = [[int(r['script_id']), r['character']] for r in rows]
    masks = torch.empty(len(rows), 1, 256, 256, dtype=torch.uint8)
    mismatch = []
    for i, row in enumerate(rows):
        face = FontFace(row['font_path'], int(row['font_face'] or 0))
        full = render_base_mask(row['character'], face, output_size=256)
        masks[i, 0] = torch.from_numpy(np.asarray(full, dtype=np.uint8))
        stored = Image.open(store / row['path'])
        if stored.size != (64, 64): raise ValueError('Store image is not 64px: ' + row['path'])
        redown = full.resize((64, 64), Image.Resampling.BOX)
        if np.asarray(redown, dtype=np.uint8).tobytes() != np.asarray(stored.convert('L'), dtype=np.uint8).tobytes():
            mismatch.append((int(row['script_id']), row['character']))
        if (i + 1) % 2000 == 0: print(f'rendered {i + 1}/{len(rows)}', flush=True)
    if mismatch:
        raise ValueError(f'{len(mismatch)} rasters disagree with the stored 64px cache: {mismatch[:5]}')
    index_sha = hashlib.sha256((store / 'index.csv').read_bytes()).hexdigest()
    torch.save(dict(keys=keys, masks=masks, source_index_sha256=index_sha), out)
    print('RASTERS_READY', out, len(keys), flush=True)


if __name__ == '__main__':
    main()
