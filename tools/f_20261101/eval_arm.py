"""Unified acceptance evaluation for one checkpoint (the F-plan gate).

Emits a single JSON covering the four acceptance axes:
  content   — SSIM / LPIPS / Ink-IoU / frag on a held-out panel
  ocr       — Top-1 on the generated panel (+ GT sanity)
  ink       — stroke width p50/p90 and width_jump
  writer    — swap direction / amplitude / OCR (via tools.diag.t3_writer_swap)

Never mixes panels or metric implementations: content comes from the official
`src.unet.io.evaluate` kernel, writer from the T3 tool, both on one checkpoint.
"""
import argparse
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, '.')
from src.unet.io import load, evaluate          # noqa: E402
from tools.diag.common import ocr_setup, ocr_score   # noqa: E402


def _png_dir(out_dir, step):
    """in_mem_eval writes eval_samples_ctrl/step{step:07d}/eval200fix/g{i}.png."""
    base = Path(out_dir) / 'eval_samples_ctrl'
    for stem in (f'step{step:07d}', f'step{step}'):
        for leaf in ('eval200fix', 'seen', 'strict'):
            cand = base / stem / leaf
            if cand.is_dir() and _gen_pngs(cand):
                return cand, leaf
    return None, None


def _gen_pngs(d):
    """Generated samples only — `g*.png` would also match `gt*.png`."""
    return sorted((p for p in d.glob('g*.png') if p.stem[1:].isdigit()),
                  key=lambda p: int(p.stem[1:]))


def content_eval(model, cfg, step, panel, out_dir, device):
    rows = [r for r in __import__('csv').DictReader(open(panel, encoding='utf-8-sig'))]
    cfg_run = replace(cfg, eval_csv=panel, output=str(out_dir))
    with torch.no_grad():
        result = evaluate(model, cfg_run, step, str(out_dir), use_ema=True)
    met = result.get('eval200fix', result)
    keep = {k: met.get(k) for k in
            ('n', 'ssim', 'mse', 'lpips', 'ink_iou', 'skel_iou', 'frag', 'hole')}
    return keep, rows


def ocr_and_ink(png_dir, rows, device, limit=None):
    """OCR Top-1 on generated PNGs + stroke-width stats on the same images."""
    from tools.p125.quality import defect_features, load_gray
    pngs = _gen_pngs(png_dir)
    if limit:
        pngs = pngs[:limit]
    chars = [(rows[i].get('char') or rows[i].get('character')) for i in range(len(pngs))]
    net, char2idx = ocr_setup(device)
    ocr = ocr_score(net, [str(p) for p in pngs], chars, char2idx, device=device)
    widths = []
    for p in pngs:
        try:
            widths.append(defect_features(load_gray(str(p))))
        except Exception:
            continue
    ink = {}
    if widths:
        for key in ('width_p50', 'width_p90', 'width_jump', 'ink_ratio', 'fragment_share'):
            vals = [w[key] for w in widths if key in w]
            ink[key] = round(sum(vals) / len(vals), 4) if vals else None
    return ocr, ink, len(pngs)


def writer_eval(checkpoint, csv_path, out_dir, tag, device, chars, writers, seeds):
    """Shell out to the T3 tool so there is exactly one swap implementation."""
    cmd = [sys.executable, '-u', '-m', 'tools.diag.t3_writer_swap',
           '--checkpoint', checkpoint, '--csv', csv_path,
           '--chars', str(chars), '--writers', str(writers), '--seeds', str(seeds),
           '--tag', tag, '--device', device, '--out', str(out_dir)]
    print('[eval] ' + ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    report = Path(out_dir) / f't3_{tag}' / 't3_report.json'
    return json.loads(report.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--panel', default='assets/u100_pretrain_20261024/validation_val100.csv')
    ap.add_argument('--csv', default='assets/p125_20261030/stage_p2.csv',
                    help='writer-swap grid source')
    ap.add_argument('--out', default='assets/f_20261101/eval')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--chars', type=int, default=30)
    ap.add_argument('--writers', type=int, default=10)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--skip-writer', action='store_true')
    ap.add_argument('--off-sites', default='',
                    help='comma list of injection sites to disable for THIS eval only '
                         '(e.g. "0,1"); the checkpoint on disk is not modified')
    ap.add_argument('--use-ckpt-decoder', action='store_true',
                    help='decode with the fine-tuned VAE decoder stored in the checkpoint '
                         '(D1/D2 joint arms) instead of the stock sd-vae decoder')
    ap.add_argument('--decoder-from', default='',
                    help='load the decoder from this other checkpoint while keeping the '
                         'model from --checkpoint (cross-combination eval, e.g. D1 decoder '
                         'x D3 U-Net)')
    a = ap.parse_args()

    out_dir = Path(a.out) / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    model, cfg, ck = load(a.checkpoint, a.device, use_ema=True)
    step = int(ck['train_steps'])
    disabled = ck.get('disabled_injection_sites', [])
    off = [int(s) for s in a.off_sites.split(',') if s.strip() != '']
    if off:
        for s in off:
            model.style_injections[s].gain.fill_(0.0)
        print(f'[eval] inference-only site disable: {off}', flush=True)
    print(f'[eval] {a.tag} step={step} ckpt_disabled_sites={disabled}', flush=True)

    decoder_override = None
    if a.use_ckpt_decoder or a.decoder_from:
        if a.decoder_from:
            donor = torch.load(a.decoder_from, map_location='cpu', weights_only=False)
            dec = donor.get('decoder_ema') or donor.get('decoder')
            if dec is None:
                raise SystemExit(f'{a.decoder_from} carries no joint decoder state')
            decoder_override = f'{a.decoder_from}'
        else:
            dec = ck.get('decoder_ema') or ck.get('decoder')
            if dec is None:
                raise SystemExit(f'{a.checkpoint} carries no joint decoder state')
            decoder_override = 'decoder_ema' if ck.get('decoder_ema') is not None else 'decoder'
        from src.eval.decoder import _get_vae
        vae = _get_vae(a.device, str(cfg.vae_path or ''),
                       float(getattr(cfg, 'vae_shift_factor', 0.0) or 0.0))
        base = getattr(vae, '_base', vae)
        # ck['decoder'] is the decoder's own state_dict (no 'decoder.' prefix).
        base.decoder.load_state_dict({k: v for k, v in dec.items()}, strict=True)
        print(f'[eval] decoder override loaded ({decoder_override})', flush=True)

    content, rows = content_eval(model, cfg, step, a.panel, out_dir, a.device)
    png_dir, panel_kind = _png_dir(out_dir, step)
    ocr = ink = None
    n_png = 0
    if png_dir is not None:
        ocr, ink, n_png = ocr_and_ink(png_dir, rows, a.device)
    del model
    torch.cuda.empty_cache()

    writer = None
    if not a.skip_writer:
        writer = writer_eval(a.checkpoint, a.csv, out_dir, a.tag, a.device,
                             a.chars, a.writers, a.seeds)

    report = dict(tag=a.tag, checkpoint=a.checkpoint, step=step,
                  disabled_injection_sites=disabled, off_sites_this_eval=off,
                  decoder_override=decoder_override, panel=a.panel,
                  panel_kind=panel_kind, n_png=n_png,
                  content=content, ocr=ocr, ink=ink,
                  writer=None if writer is None else {
                      k: writer.get(k) for k in ('grid', 'ocr', 'direction',
                                                 'amplitude', 'n_pairs', 'verdict_hint')})
    (out_dir / 'eval.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print('[eval] wrote', out_dir / 'eval.json', flush=True)
    print('[eval]', json.dumps({k: report[k] for k in ('tag', 'step', 'content')},
                               ensure_ascii=False), flush=True)
    if report['writer']:
        print('[eval] writer', json.dumps(report['writer'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
