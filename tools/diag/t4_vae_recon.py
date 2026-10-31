"""T4 — VAE round-trip fidelity on real calligraphy (CPU only).

Question (plan §T4): do glyph/detail errors live in the VAE or the generator?
  * Take real GT images and run GT -> encoder -> latent -> decoder -> recon.
  * Score BOTH the GT and the reconstruction with the SAME OCR net and the
    SAME ink metric block used for generations (no panel mixing).
  * `--use-shard-latent` decodes the prebuilt training latents instead, which
    removes the encoder from the loop and isolates decoder-only damage.

Verdict rule (plan): if GT reconstruction stays clean while generations are
wrong, optimize the generator; if reconstructions themselves drop thin strokes
or merge parts (ink_iou / frag / hole degrade, OCR drops), the decoder is the
suspect. The report prints raw numbers plus a threshold-based hint — the final
call stays with the plan's Phase-3 decision.
"""
import sys
from pathlib import Path
import statistics as st

import numpy as np
import torch

sys.path.insert(0, '.')
from tools.diag.common import (base_args, load_model, read_rows, pin,  # noqa: E402
                               shard_latents, load_vae, encode_images,
                               decode_latents, gt_image_tensor, ocr_setup,
                               ocr_score, set_metrics, write_json, slot_of)


def to_hwc01(t3):
    """(3,H,W) in [-1,1] -> (H,W,3) in [0,1] (official metric layout)."""
    return ((t3.permute(1, 2, 0).numpy() + 1.0) / 2.0).clip(0, 1)


def save_png(hw3, path):
    from PIL import Image
    Image.fromarray((hw3[..., 0] * 255).astype(np.uint8)).save(path)


def main():
    a = base_args(extra=[('--use-shard-latent', dict(action='store_true')),
                         ('--size', dict(type=int, default=256))])
    pin(a.threads)
    out_dir = Path(a.out) / 't4'
    png_dir = out_dir / 'png'
    png_dir.mkdir(parents=True, exist_ok=True)
    _model, cfg, _ck = load_model(a.checkpoint, 'cpu')   # config carrier only
    del _model
    vae, sf, shift = load_vae(cfg, a.device)
    # stage CSVs have no `writer_trusted` column: absence means "training row"
    rows = [r for r in read_rows(a.csv)
            if r.get('writer_trusted') is None or int(r['writer_trusted']) == 1][:a.n]
    img_ids = [int(r['img_id']) for r in rows]
    shard = shard_latents(a.shards, img_ids)
    mode = 'shard_latent' if a.use_shard_latent else 'encoder_roundtrip'
    print(f'[t4] mode={mode} rows={len(rows)} shard={len(shard)} vae={cfg.vae_path} sf={sf}', flush=True)

    net, char2idx = ocr_setup(a.device)
    per_row, kept = [], []
    for r in rows:
        iid = int(r['img_id'])
        try:
            gt3 = gt_image_tensor(r['image_path'], a.size)
        except Exception as e:
            print(f'[t4] skip {iid}: gt missing ({e})', flush=True)
            continue
        if a.use_shard_latent:
            if iid not in shard:
                continue
            lat = shard[iid][None]
        else:
            with torch.no_grad():
                lat = encode_images(vae, gt3[None], sf, shift)
        with torch.no_grad():
            recon3 = decode_latents(vae, lat, sf, shift, batch=1)[0]
        gt01, rc01 = to_hwc01(gt3), to_hwc01(recon3)
        met = set_metrics(rc01[None], gt01[None], 1)
        g_png, r_png = png_dir / f'gt{iid}.png', png_dir / f'recon{iid}.png'
        save_png(gt01, g_png)
        save_png(rc01, r_png)
        flat = {k: (round(v, 5) if isinstance(v, (int, float)) else None)
                for k, v in met.items()}
        per_row.append(dict(img=iid, char=r['character'], slot=slot_of(r),
                            metrics=flat, gt_png=str(g_png), recon_png=str(r_png)))
        kept.append(dict(char=r['character'], gt_png=str(g_png), recon_png=str(r_png)))
        if len(per_row) % 25 == 0:
            print(f'[t4] {len(per_row)} reconstructions done', flush=True)

    if not per_row:
        raise SystemExit('[t4] no reconstructions produced — check image_path / shard ids')
    chars = [k['char'] for k in kept]
    ocr_gt = ocr_score(net, [k['gt_png'] for k in kept], chars, char2idx)
    ocr_rc = ocr_score(net, [k['recon_png'] for k in kept], chars, char2idx)

    def col(key):
        vals = [p['metrics'][key] for p in per_row
                if isinstance(p['metrics'].get(key), (int, float))]
        return round(st.fmean(vals), 5) if vals else None

    ocr_drop = (ocr_gt['top1'] - ocr_rc['top1']) if (ocr_gt and ocr_rc) else None
    mm = {k: col(k) for k in ('ssim', 'mse', 'ink_iou', 'skel_iou', 'frag', 'hole')}
    # Verdict: OCR is the primary gate when the vocab covers the sample; if the
    # sampled glyphs are mostly out-of-vocab, fall back to the ink metrics.
    if ocr_drop is not None:
        clean = ocr_drop < 5 and (mm['ink_iou'] or 0) > 0.9
        basis = f'ocr_top1_drop={ocr_drop} ink_iou={mm["ink_iou"]}'
    else:
        clean = (mm['ink_iou'] or 0) > 0.95 and (mm['skel_iou'] or 0) > 0.8
        basis = f'ink_iou={mm["ink_iou"]} skel_iou={mm["skel_iou"]} (OCR out-of-vocab)'
    summary = {
        'mode': mode,
        'vae': cfg.vae_path,
        'n': len(per_row),
        'ocr_gt': ocr_gt,
        'ocr_recon': ocr_rc,
        'ocr_top1_drop': ocr_drop,
        'mean_metrics': mm,
        'verdict_hint': ('reconstruction clean -> optimize generator'
                         if clean else 'reconstruction loses content -> decoder suspect'),
        'verdict_basis': basis,
    }
    write_json(out_dir / 't4_report.json', {'summary': summary, 'per_row': per_row[:80]})
    print('=== T4 summary ===', flush=True)
    for k, v in summary.items():
        if k != 'per_row':
            print(f'  {k}: {v}', flush=True)


if __name__ == '__main__':
    main()
