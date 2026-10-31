"""T3 — Writer-swap generation test (CPU only).

Protocol (plan §T3): fix character, STD, script and noise; vary ONLY the writer
id; measure whether the generated style moves the way real writers differ.

Grid: `--chars` shared characters x `--writers` writer slots x `--seeds` noise
seeds. Noise is constructed here as base_noise(char, seed) and reused for every
writer of that (char, seed) cell, so any difference between writers comes from
the condition alone.

Three readouts (definitions documented, so they are reproducible):
  * Direction — for every writer pair (a,b) and every character c where both
    writers have real samples, take the generated latent difference
    d_gen = v[c,a] - v[c,b] and the real one d_true = u[c,a] - u[c,b]
    (u = mean of that writer's real glyph latents for c, from the training
    shards). Report the mean cosine. ~0 means the swap direction is unrelated
    to real writer differences.
  * Amplitude — mean ||d_gen|| / ||d_true||. ~0 means the swap barely moves the
    latent; >>1 means the condition overshoots.
  * Content stability — OCR Top-1 of the generated image against the target
    character (same OCR net and preprocessing as the panel tooling).

Everything runs on CPU through the official `make_eval_cache` / `sample_latents`
so the conditioning path is bit-identical to the training-time evaluator.
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path
import statistics as st

import numpy as np
import torch

sys.path.insert(0, '.')
from tools.diag.common import (base_args, load_model, read_rows, pin,  # noqa: E402
                               shard_latents, font_map, eval_args_of,
                               diffusion_of, ocr_setup, ocr_score,
                               load_vae, decode_latents, write_json)


def pick_grid(rows, n_chars, n_writers):
    """Characters seen with many writers x the most frequent writer slots."""
    by_char = defaultdict(dict)      # char -> {slot: row}
    slot_freq = Counter()
    for r in rows:
        if int(r.get('writer_trusted', 0)) != 1:
            continue
        slot = int(r['writer_slot'])
        ch = r['character']
        slot_freq[slot] += 1
        by_char[ch].setdefault(slot, r)
    writers = [s for s, _ in slot_freq.most_common(n_writers)]
    need = max(2, min(3, n_writers))
    cand = [(ch, m) for ch, m in by_char.items() if len(set(m) & set(writers)) >= need]
    cand.sort(key=lambda kv: -len(set(kv[1]) & set(writers)))
    chars = [ch for ch, _ in cand[:n_chars]]
    return chars, writers, by_char


def main():
    a = base_args(extra=[('--chars', dict(type=int, default=30)),
                         ('--writers', dict(type=int, default=10)),
                         ('--seeds', dict(type=int, default=3)),
                         ('--csv-out', dict(default='')),
                         ('--no-ocr', dict(action='store_true'))])
    pin(a.threads)
    out_dir = Path(a.out) / 't3'
    png_dir = out_dir / 'png'
    png_dir.mkdir(parents=True, exist_ok=True)
    model, cfg, ck = load_model(a.checkpoint)
    args = eval_args_of(cfg)
    diffusion = diffusion_of(args)
    srow = font_map(cfg)
    print(f'[t3] model={a.checkpoint} step={ck["train_steps"]} cfg={args.eval_cfg} '
          f'steps={args.eval_steps}', flush=True)

    rows_all = read_rows(a.csv)
    chars, writers, by_char = pick_grid(rows_all, a.chars, a.writers)
    # The remap tables are keyed by RAW calligrapher ids, so a synthetic row for
    # writer slot w must reuse a real raw id that maps to w.
    raw_of_slot = {}
    for r in rows_all:
        if int(r.get('writer_trusted', 0)) == 1:
            raw_of_slot.setdefault(int(r['writer_slot']), int(r['calligrapher_id']))
    print(f'[t3] chars={len(chars)} writers={len(writers)} seeds={a.seeds} '
          f'raw_of_slot={len(raw_of_slot)}', flush=True)

    # ---- build the swap grid CSV (writer varies, everything else fixed)
    grid = []       # (char, writer, seed) -> row dict
    for ch in chars:
        for w in writers:
            gt = by_char[ch].get(w)
            if gt is None or w not in raw_of_slot:
                continue          # no real sample for this (char, writer): skip cell
            for s in range(a.seeds):
                row = dict(gt)
                row['calligrapher_id'] = str(raw_of_slot[w])
                row['writer_slot'] = str(w)
                grid.append((ch, w, s, row))
    if not grid:
        raise SystemExit('[t3] empty grid — no char has real samples for >=1 writer')
    csv_path = Path(a.csv_out or (out_dir / 'swap_grid.csv'))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    import csv as _csv
    fields = sorted({k for _, _, _, r in grid for k in r})
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = _csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for _, _, _, r in grid:
            w.writerow(r)
    print(f'[t3] grid rows={len(grid)} csv={csv_path}', flush=True)

    # ---- official cache (conditioning path identical to the evaluator)
    from src.eval.dataset import make_eval_cache
    _soft = getattr(args, 'std_soft_root', '') or ''
    cache = make_eval_cache(
        str(csv_path), None, None, 256, len(grid), 8, 4, 0.18215,
        skel_latent_shards_dir=(None if _soft else getattr(args, 'latent_shards_dir', '')),
        callig_id_map=None,
        callig_script_map=None,
        std_soft_root=_soft or None,
        std_soft_sigma=float(getattr(args, 'std_soft_eval_sigma', 0.65)))
    # MatrixUNet consumes only y_callig / y_char / g / y_script, so the swap is
    # expressed directly: writer slot as the callig id, glyph_id as y_char, and
    # the font_remap'ed script as the third hier axis (the sampler reads
    # y_script from there). No remap-table detour, nothing depends on the
    # trusted CSV's raw calligrapher ids.
    def _int(v, d=0):
        try:
            return int(v)
        except (TypeError, ValueError):
            return d

    cache['conds'] = [(w, _int(r.get('glyph_id'))) for _ch, w, _s, r in grid]
    cache['hier_conds'] = (
        torch.tensor([w for _ch, w, _s, _r in grid], dtype=torch.long),
        torch.tensor([_int(r.get('pair_id')) for _ch, _w, _s, r in grid], dtype=torch.long),
        torch.tensor([srow(_int(r['script_id'])) for _ch, _w, _s, r in grid], dtype=torch.long),
    )
    print(f'[t3] cache keys={sorted(cache)} n={cache["n"]}', flush=True)

    # ---- fixed noise per (char, seed), shared across writers
    import zlib
    noise = torch.zeros(len(grid), 4, 32, 32)
    for i, (ch, _w, s, _r) in enumerate(grid):
        seed_i = (zlib.crc32(ch.encode('utf-8')) + 1000 * s) % (2 ** 31)
        gen = torch.Generator().manual_seed(seed_i)
        noise[i] = torch.randn(4, 32, 32, generator=gen)
    print('[t3] noise built (shared across writers per char/seed)', flush=True)

    from src.eval.sampler import sample_latents
    import time as _time
    _t0 = _time.time()
    with torch.no_grad():
        latents = sample_latents(model, diffusion, noise, cache['conds'],
                                 float(getattr(args, 'eval_cfg', 1.0)),
                                 int(getattr(args, 'in_mem_eval_batch', 8)),
                                 torch.device('cpu'),
                                 skel=cache.get('skels_latent'), seed=0,
                                 hier_conds=cache.get('hier_conds'))
    latents = latents.float()
    _dt = _time.time() - _t0
    print(f'[t3] sampled {tuple(latents.shape)} in {_dt / 60:.1f} min '
          f'({_dt / max(len(grid), 1):.2f} s/img)', flush=True)

    # ---- decode + save + OCR
    vae, sf, shift = load_vae(cfg)
    images = decode_latents(vae, latents, sf, shift, batch=4)
    for i, (ch, w, s, _r) in enumerate(grid):
        from PIL import Image
        arr = ((images[i].mean(0).numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(arr).save(png_dir / f'g{i:04d}_{ch}_{w}_s{s}.png')
    print(f'[t3] decoded {len(grid)} images', flush=True)

    ocr = None
    if not a.no_ocr:
        net, char2idx = ocr_setup()
        pngs = [str(png_dir / f'g{i:04d}_{ch}_{w}_s{s}.png')
                for i, (ch, w, s, _r) in enumerate(grid)]
        ocr = ocr_score(net, pngs, [ch for ch, _, _, _ in grid], char2idx)
        print(f'[t3] OCR {ocr}', flush=True)

    # ---- aggregate: generated per (char,writer) mean latent
    gen_mean = defaultdict(list)
    for i, (ch, w, _s, _r) in enumerate(grid):
        gen_mean[(ch, w)].append(latents[i].flatten())
    gen_mean = {k: torch.stack(v).mean(0) for k, v in gen_mean.items()}

    # ---- real per (char,writer) mean latent from the training shards
    gt_ids, gt_key = [], {}
    for ch in chars:
        for w in writers:
            r = by_char[ch].get(w)
            if r is None:
                continue
            iid = int(r['img_id'])
            gt_ids.append(iid)
            gt_key[(ch, w)] = iid
    gt_lat = shard_latents(a.shards, gt_ids)
    true_mean = {}
    for k, iid in gt_key.items():
        if iid in gt_lat:
            true_mean[k] = gt_lat[iid].flatten()

    # ---- direction / amplitude over writer pairs, per character
    dirs, amps, pairs_n = [], [], 0
    per_char = {}
    for ch in chars:
        ws = [w for w in writers if (ch, w) in gen_mean and (ch, w) in true_mean]
        if len(ws) < 2:
            continue
        c_dirs, c_amps = [], []
        for i in range(len(ws)):
            for j in range(i + 1, len(ws)):
                wa, wb = ws[i], ws[j]
                dg = gen_mean[(ch, wa)] - gen_mean[(ch, wb)]
                dt = true_mean[(ch, wa)] - true_mean[(ch, wb)]
                ng, nt = float(dg.norm()), float(dt.norm())
                if ng < 1e-8 or nt < 1e-8:
                    continue
                c_dirs.append(float((dg / ng) @ (dt / nt)))
                c_amps.append(ng / nt)
                pairs_n += 1
        if c_dirs:
            per_char[ch] = dict(direction=round(st.fmean(c_dirs), 4),
                                amplitude=round(st.fmean(c_amps), 4),
                                pairs=len(c_dirs))
            dirs += c_dirs
            amps += c_amps

    report = {
        'checkpoint': a.checkpoint,
        'step': int(ck['train_steps']),
        'grid': dict(chars=len(chars), writers=len(writers), seeds=a.seeds,
                     rows=len(grid)),
        'ocr': ocr,
        'direction': round(st.fmean(dirs), 4) if dirs else None,
        'amplitude': round(st.fmean(amps), 4) if amps else None,
        'n_pairs': pairs_n,
        'per_char': per_char,
        'verdict_hint': None,
    }
    if dirs:
        d = st.fmean(dirs)
        if d > 0.5:
            hint = 'writer swap moves in the RIGHT direction'
        elif d > 0.1:
            hint = ('writer swap direction is weak (<0.5) — writer identity is '
                    'not being transferred')
        else:
            hint = ('writer swap direction is essentially random — writer '
                    'conditioning is inert at generation time')
        report['verdict_hint'] = hint
    write_json(out_dir / 't3_report.json', report)
    print('\n=== T3 summary ===', flush=True)
    for k in ('grid', 'ocr', 'direction', 'amplitude', 'n_pairs', 'verdict_hint'):
        print(f'  {k}: {report[k]}', flush=True)


if __name__ == '__main__':
    main()
