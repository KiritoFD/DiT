"""F4: of the real per-stroke writer differences, which ones did the model learn?

Everything before this measured REAL calligraphy. This measures the gap to what
the local 114M U-Net actually produces, using the SAME event extractor on
generated images -- so the comparison is like-for-like instead of "the pictures
look similar".

Per (character, writer) it generates the glyph with fixed STD and noise, then
extracts end / junction / turn events exactly as for GT, and compares the
writer-difference VECTORS:

    direction agreement  cos( Delta^GT_AB , Delta^Gen_AB ) per event kind
    magnitude ratio      || Delta^Gen || / || Delta^GT ||

A model that reproduces a writer's mean weight but not their stroke ends will
show a high cosine on `turn` and a low one on `end`. That is the actionable
statement the plan asks for -- which specific local habit is missing -- instead
of a single "style score".
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

E_KINDS = ['end', 'junction', 'turn']
GEO = ['asym_ud', 'asym_lr', 'grad_mag', 'turn_angle', 'end_area', 'end_elong',
       'n_branches', 'orient_peak']
INK = ['mass', 'ink_frac']
ALLF = GEO + INK
STD_DIR = 'exp-std/data/std_soft_glyphs_v1'
SCRIPT_CODE = {'楷': 'kai', '行': 'xing', '隶': 'li'}


def std_path(script, character):
    """STD soft glyph path; the bank names files as U+%05X (5 hex digits)."""
    p = ROOT / STD_DIR / SCRIPT_CODE[script] / f'U+{ord(character):05X}.png'
    return p if p.exists() else None


def event_means(img_arr, rng, per_kind=24):
    """Run the shared extractor on an in-memory image -> {kind: feature mean}."""
    from PIL import Image
    from tools.stroke_interaction.features import (
        glyph_features, extract_events, structure_features)
    im = Image.fromarray((np.clip(img_arr, 0, 1) * 255).astype(np.uint8)).convert('L')
    arr = np.array(im, np.float32) / 255.0
    ink = 1.0 - arr if arr.mean() > 0.5 else arr
    binary = (ink > 0.5).astype(np.uint8)
    if binary.sum() < 32:
        return None
    events, _, _ = extract_events(ink, binary, rng, per_kind=per_kind)
    out = defaultdict(list)
    for e in events:
        out[e['kind']].append([e.get(k, 0.0) for k in ALLF])
    if not out:
        return None
    return {k: np.array(v, np.float32).mean(0) for k, v in out.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--csv', default='exp-std/csv/train_top10_noaug.csv')
    ap.add_argument('--out', default='assets/stroke_writer_20261026')
    ap.add_argument('--chars', type=int, default=40)
    ap.add_argument('--per-kind', type=int, default=20)
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--cfg', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--raw', action='store_true',
                    help='load raw weights (legacy checkpoints have no verified EMA)')
    ap.add_argument('--vae-path',default='',help='Audited decoder-only override')
    ap.add_argument('--latent-cache',default='',help='Reuse exactly the same generated latent across decoders')
    a = ap.parse_args(argv)

    import csv
    from PIL import Image
    from src.unet.io import load, diffusion
    from src.eval.sampler import sample_latents

    model, config, ck = load(a.ckpt, 'cuda', use_ema=not a.raw)
    process = diffusion(config)
    print(f'[f4] loaded {a.ckpt} step={ck.get("train_steps", "?")}', flush=True)

    rows = list(csv.DictReader(open(ROOT / a.csv, encoding='utf-8')))
    import os
    n_missing = sum(1 for r in rows if not (ROOT / r['image_path']).exists())
    rows = [r for r in rows if (ROOT / r['image_path']).exists()]
    print(f'[f4] csv {a.csv}: {len(rows)} rows with images on disk '
          f'({n_missing} dropped as missing)', flush=True)
    # pick characters with the widest writer coverage, so a writer difference is
    # estimable from several real specimens
    by_char = defaultdict(set)
    for r in rows:
        by_char[(r['script'], r['character'])].add(r['calligrapher'])
    cand = sorted(by_char.items(), key=lambda kv: -len(kv[1]))[:a.chars]
    print(f'[f4] {len(cand)} characters, '
          f'writer coverage {min(len(v) for _, v in cand)}..{max(len(v) for _, v in cand)}',
          flush=True)

    # Condition ids must match the TRIPLE TABLES this checkpoint was trained on,
    # not the raw CSV ids: writer 10 entries (table has 11 rows = +1 null) and
    # script {0,3,4} -> {0,1,2} (table has 4 rows). Passing the raw script id 4
    # for 隶 is exactly what triggered the device-side assert.
    # ckpt path is <run>/checkpoints/<step>.pt, so the run dir is parents[1]
    ck_dir = (ROOT / a.ckpt).parents[1]
    cfg = json.loads((ck_dir / 'resolved_config.json').read_text(encoding='utf-8'))
    callig_remap = json.loads((ROOT / cfg['callig_remap_json']).read_text(encoding='utf-8'))
    font_remap = json.loads((ROOT / cfg['font_remap_json']).read_text(encoding='utf-8'))
    print(f'[f4] config data_csv={cfg["data_csv"]}', flush=True)

    def writer_id(row):
        a_ = callig_remap.get(str(row['calligrapher_id']))
        b_ = font_remap.get(str(row['script_id']))
        if a_ is None or b_ is None:
            return None, None
        return int(a_), int(b_)

    from diffusers import AutoencoderKL
    import torch.nn.functional as Fn
    vae = AutoencoderKL.from_pretrained(a.vae_path or config.vae_path, torch_dtype=torch.bfloat16,
                                        local_files_only=True).cuda().eval()
    cache=None
    if a.latent_cache:
        import hashlib
        cache=ROOT/a.latent_cache; cache.mkdir(parents=True,exist_ok=True)
        h=hashlib.sha256()
        with open(a.ckpt,'rb') as f:
            for block in iter(lambda:f.read(8<<20),b''): h.update(block)
        identity=dict(checkpoint_sha256=h.hexdigest(),ema=not a.raw,csv_sha256=hashlib.sha256(Path(a.csv).read_bytes()).hexdigest(),
                      seed=a.seed,cfg=a.cfg,actual_steps=process.num_timesteps)
        ident=cache/'identity.json'
        if ident.exists(): assert json.loads(ident.read_text())==identity, 'Latent cache protocol changed'
        else: ident.write_text(json.dumps(identity,indent=2))

    rng = np.random.default_rng(a.seed)
    gt_means = defaultdict(dict)     # (script,char) -> writer -> {kind: vec}
    for (sc, ch), writes in cand:
        for w in sorted(writes):
            rs = [r for r in rows if r['script'] == sc and r['character'] == ch
                  and r['calligrapher'] == w][:3]
            vs = []
            for r in rs:
                arr = np.array(Image.open(ROOT / r['image_path']).convert('L'),
                               np.float32) / 255.0
                m = event_means(arr, rng, a.per_kind)
                if m:
                    vs.append(m)
            if vs:
                agg = defaultdict(list)
                for m in vs:
                    for k, v in m.items():
                        agg[k].append(v)
                gt_means[(sc, ch)][w] = {k: np.mean(v, 0) for k, v in agg.items()}
    print(f'[f4] GT event means for {len(gt_means)} characters', flush=True)

    # ---- generate the same characters under each writer ----
    # Keep one representative CSV row per (script, character, writer) so the
    # writer id comes from the same table the checkpoint was trained with.
    rep_row = {}
    for r in rows:
        k = (r['script'], r['character'], r['calligrapher'])
        rep_row.setdefault(k, r)

    gen_means = defaultdict(dict)
    skipped_no_std = 0
    skipped_no_id = 0
    for (sc, ch), writes in cand:
        std_p = std_path(sc, ch)
        if std_p is None:
            skipped_no_std += 1
            continue
        s = np.array(Image.open(std_p).convert('L'), np.float32) / 255.0
        g64 = torch.from_numpy(s)[None, None]
        pending = []
        for w in sorted(writes):
            row = rep_row.get((sc, ch, w))
            wid, sid = writer_id(row) if row is not None else (None, None)
            if wid is None:
                skipped_no_id += 1
                continue
            pending.append((w,wid,sid))
        if not pending:
            continue
        # Same per-writer initial noise, same sampler, batched forward. Previous
        # implementation sampled every writer individually and underfed GPU.
        wid_list=[v[1] for v in pending]; sid_list=[v[2] for v in pending]
        conds = [(wid,0) for wid in wid_list]
        # hier_conds = [calligrapher ids, char ids, script ids], remapped
        hier = [wid_list, [0]*len(pending), sid_list]
        noise = torch.randn(1,4,32,32,generator=torch.Generator().manual_seed(a.seed)).repeat(len(pending),1,1,1)
        with torch.no_grad():
            cache_file=cache/f'{SCRIPT_CODE[sc]}_{ord(ch):05X}.pt' if cache else None
            if cache_file is not None and cache_file.exists():
                saved=torch.load(cache_file,map_location='cpu',weights_only=False)
                assert saved['pending']==pending
                torch.testing.assert_close(saved['g'],g64,rtol=0,atol=0)
                lat=saved['latents']
            else:
                lat = sample_latents(model,process,noise,conds,a.cfg,len(pending),'cuda',
                                     skel=g64.repeat(len(pending),1,1,1),seed=a.seed,hier_conds=hier)
                if cache_file is not None: torch.save(dict(latents=lat,pending=pending,g=g64),cache_file)
            x = vae.decode((lat.cuda().float()/config.vae_scaling_factor+config.vae_shift_factor).to(torch.bfloat16)).sample
        for j,(w,wid,sid) in enumerate(pending):
            img = ((x.float().clamp(-1, 1) + 1) / 2)[j, 0].cpu().numpy()
            m = event_means(img, rng, a.per_kind)
            if m:
                gen_means[(sc, ch)][w] = m
    print(f'[f4] generated for {len(gen_means)} characters '
          f'(skipped: no STD={skipped_no_std}, no writer id={skipped_no_id})', flush=True)

    # ---- compare writer-difference vectors ----
    rep = {'ckpt': a.ckpt, 'n_chars_gt': len(gt_means), 'n_chars_gen': len(gen_means),
           'decoder':a.vae_path or config.vae_path,'latent_cache':a.latent_cache,
           'by_kind': {}, 'per_feature': {}}
    acc = defaultdict(list)
    acc_feat = defaultdict(lambda: defaultdict(list))
    for key in set(gt_means) & set(gen_means):
        ws = sorted(set(gt_means[key]) & set(gen_means[key]))
        if len(ws) < 2:
            continue
        for i in range(len(ws)):
            for j in range(i + 1, len(ws)):
                for k in E_KINDS:
                    g1 = gt_means[key][ws[i]].get(k)
                    g2 = gt_means[key][ws[j]].get(k)
                    n1 = gen_means[key][ws[i]].get(k)
                    n2 = gen_means[key][ws[j]].get(k)
                    if g1 is None or g2 is None or n1 is None or n2 is None:
                        continue
                    dGT, dGen = g1 - g2, n1 - n2
                    nGT, nGen = np.linalg.norm(dGT), np.linalg.norm(dGen)
                    if nGT < 1e-9:
                        continue
                    cos = float(dGT @ dGen / (nGT * max(nGen, 1e-12)))
                    acc[k].append((cos, nGen / nGT))
                    for fi, fname in enumerate(ALLF):
                        a_, b_ = dGT[fi], dGen[fi]
                        acc_feat[k][fname].append(
                            float(np.sign(a_) == np.sign(b_)) if abs(a_) > 1e-9 else np.nan)
    for k, vals in acc.items():
        cos = np.array([v[0] for v in vals])
        mag = np.array([v[1] for v in vals])
        rep['by_kind'][k] = {
            'n_pairs': len(vals),
            'mean_cosine': round(float(cos.mean()), 4),
            'frac_positive': round(float((cos > 0).mean()), 4),
            'median_magnitude_ratio': round(float(np.median(mag)), 4),
        }
        print(f'[f4] {k:9s} n={len(vals):4d} cos={cos.mean():+.4f} '
              f'pos={float((cos>0).mean()):.2f} mag_ratio={np.median(mag):.3f}', flush=True)
    for k, fam in acc_feat.items():
        rep['per_feature'][k] = {
            f: round(float(np.nanmean(v)), 4) for f, v in fam.items() if np.any(~np.isnan(v))}
        top = sorted(rep['per_feature'][k].items(), key=lambda kv: kv[1])[:4]
        print(f'[f4] {k:9s} worst-reproduced features: {top}', flush=True)

    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    p = out / 'gt_vs_generated.json'
    p.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[f4] wrote {p}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
