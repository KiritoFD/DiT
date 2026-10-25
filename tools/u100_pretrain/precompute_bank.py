"""Offline GPU augmentation -> FP16 CPU bank. Training never runs geometry kernels.

Same contract as the v90 precompute tool, relocated so the 4090 checkout does
not need the local-only experiments package. Clean branch renders at sigma .65
through StdSoftGlyphStore over the bound STD store index.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.std_shape_aug import StdShapeAugment
from src.utils.std_soft_glyph import StdSoftGlyphStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', required=True)
    parser.add_argument('--store', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--variants', type=int, default=8)
    parser.add_argument('--batch', type=int, default=64)
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists(): raise FileExistsError(output)
    torch.manual_seed(90)
    source = torch.load(args.raw, map_location='cpu', weights_only=True, mmap=True)
    store = StdSoftGlyphStore(args.store, sigma_min=.65, sigma_max=.65, eval_sigma=.65)
    keys = source['keys']
    clean = torch.stack([store.get(int(s), c, sigma=.65) for s, c in keys])
    bank = torch.empty(len(keys), 3, args.variants, 1, 64, 64, dtype=torch.float16)
    augmenter = StdShapeAugment('strong')
    start = time.monotonic(); fallback = total = 0
    for variant in range(args.variants):
        for tier in (1, 2, 3):
            for offset in range(0, len(keys), args.batch):
                end = min(offset + args.batch, len(keys))
                value, stats = augmenter(source['masks'][offset:end].cuda(), clean[offset:end].cuda(), force_tier=tier)
                bank[offset:end, tier - 1, variant] = value.cpu().half()
                fallback += int(stats['fallback'].sum()); total += len(value)
            print(json.dumps(dict(variant=variant + 1, tier=tier, elapsed_s=round(time.monotonic() - start, 1),
                                  fallback=fallback, total=total)), flush=True)
    torch.save(dict(version=2, source_kind='precomputed_std_shape', keys=keys, variants=bank,
                    source_index_sha256=source['source_index_sha256'], seed=90,
                    fallback_count=fallback, total=total), output)
    print('STD_BANK_READY', output, bank.shape, flush=True)


if __name__ == '__main__':
    main()
