"""T1 — Writer conditioning timeline on fixed latents (CPU only).

Protocol (experiment plan §T1, the most important diagnosis):
  * fix the real glyph latent z0, the STD mask, the script id and the noise;
  * interpolate z_t = (1-t) z0 + t eps at t in {0.10,0.25,0.40,0.60,0.80,0.95};
  * on the SAME z_t run the FM objective with the correct writer, a wrong
    writer, and the null writer (same eps for every conditioning);
  * report  dL_wrong(t) = L(wrong)-L(correct)  and  dL_null(t) = L(null)-L(correct).

Interpretation (plan): if dL_* ≈ 0 the generator largely ignores the writer;
if dL is clear but T3 shows wrong swap directions, the problem is in how the
condition acts (or data bias) rather than its magnitude.

Groupings: time t, writer bucket (Top10 vs generic slots), script id, and
character rarity (frequency inside the trusted CSV as a coverage proxy — the
CSV has no stroke-count column; complexity buckets are not fabricated).
All trusted rows are training-seen by construction; seen/unseen belongs to
T3/T4, this cut uses rarity as the closest available proxy.
"""
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
import statistics as st

import torch

sys.path.insert(0, '.')
from tools.diag.common import (base_args, load_model, read_rows, pin,  # noqa: E402
                               shard_latents, std_store_of, null_writer,
                               pick_rows, font_map, write_json)

TS = (0.10, 0.25, 0.40, 0.60, 0.80, 0.95)


def build_diffusion_for(config):
    from src.loss import flow_kwargs_from
    from src.eval.sampler import build_diffusion
    return build_diffusion(999, 'flow', flow_kwargs_from(config))


def freq_bucket(counter, character):
    c = counter.get(character, 0)
    return 'rare' if c <= 2 else ('mid' if c <= 30 else 'common')


def main():
    a = base_args()
    pin(a.threads)
    out_dir = Path(a.out) / 't1'
    out_dir.mkdir(parents=True, exist_ok=True)
    model, cfg, ck = load_model(a.checkpoint)
    diffusion = build_diffusion_for(cfg)
    null = null_writer(model)
    # CSV script ids are raw corpus ids (0=楷,3=行,4=隶); the model's script
    # embedding is indexed AFTER font_remap (see in_mem_eval §header).
    srow = font_map(cfg)
    print(f'[t1] model={a.checkpoint} step={ck["train_steps"]} null_writer={null} '
          f'font_remap={cfg.font_remap_json}', flush=True)

    rows_all = read_rows(a.csv)
    counter = Counter(r['character'] for r in rows_all)
    rows = pick_rows(rows_all, a.n)
    img_ids = [int(r['img_id']) for r in rows]
    z0s = shard_latents(a.shards, img_ids)
    store = std_store_of(cfg)
    print(f'[t1] rows={len(rows)} latents={len(z0s)}', flush=True)

    char_of = {int(r['img_id']): r['character'] for r in rows}
    script_of = {int(r['img_id']): int(r['script_id']) for r in rows}
    raw = []     # dicts: img slot seed t cond loss
    for ri, r in enumerate(rows):
        iid = int(r['img_id'])
        if iid not in z0s:
            continue
        z0 = z0s[iid][None]
        try:
            g = store.get(int(r['script_id']), r['character'])
        except Exception as e:
            print(f'[t1] skip {iid}: std store miss ({e})', flush=True)
            continue
        g = torch.as_tensor(g).float()
        g = g[None] if g.dim() == 3 else g[None, None]
        slot = int(r['writer_slot'])
        script = int(r['script_id'])
        s_id = srow(script)
        wrong = (slot + 1 + ri) % 47
        if wrong == slot:
            wrong = (slot + 7) % 47
        conds = [('correct', slot), ('wrong', wrong), ('null', null)]
        for seed_i in range(2):
            gen = torch.Generator().manual_seed(iid + seed_i)
            noise = torch.randn(z0.shape, generator=gen)
            for t in TS:
                zt = (1.0 - t) * z0 + t * noise
                tt = torch.full((1,), float(t))
                for name, w in conds:
                    kw = dict(g=g, y_callig=torch.tensor([w]),
                              y_script=torch.tensor([s_id]),
                              condition_dropped=False)
                    with torch.no_grad():
                        out = diffusion.training_losses(model, zt, tt, kw, noise)
                    loss = out['loss'] if isinstance(out, dict) else out
                    raw.append(dict(img=iid, slot=slot, seed=seed_i, t=t,
                                    cond=name, loss=round(float(loss.mean()), 6)))
        if (ri + 1) % 8 == 0 or ri + 1 == len(rows):
            print(f'[t1] rows {ri + 1}/{len(rows)} done', flush=True)

    def agg(keyfn):
        table = defaultdict(lambda: defaultdict(list))
        for d in raw:
            table[keyfn(d)][d['cond']].append(d['loss'])
        return table

    def summarize(table):
        out = {}
        for k, per in table.items():
            base = st.fmean(per['correct'])
            out[str(k)] = {
                'n_correct': len(per['correct']),
                'L_correct': round(base, 6),
                'L_wrong': round(st.fmean(per['wrong']), 6),
                'L_null': round(st.fmean(per['null']), 6),
                'dL_wrong': round(st.fmean(per['wrong']) - base, 6),
                'dL_null': round(st.fmean(per['null']) - base, 6),
            }
        return out

    by_t = summarize(agg(lambda d: d['t']))
    if not by_t:
        raise SystemExit('[t1] no usable rows — check std-soft coverage and shard ids')
    report = {
        'checkpoint': a.checkpoint,
        'step': int(ck['train_steps']),
        'csv': a.csv,
        'rows': len(rows),
        'null_writer': null,
        'by_t': by_t,
        'by_bucket': summarize(agg(lambda d: 'top10' if d['slot'] < 10 else 'generic')),
        'by_script': summarize(agg(lambda d: script_of.get(d['img'], -1))),
        'by_freq': summarize(agg(lambda d: freq_bucket(counter, char_of[d['img']]))),
        'verdict_hint': ('writer ignored' if all(abs(v['dL_wrong']) < 0.005 and
                          abs(v['dL_null']) < 0.005 for v in by_t.values())
                          else 'writer changes the velocity field; cross-check T3 directions'),
    }
    write_json(out_dir / 't1_report.json', report)
    write_json(out_dir / 't1_raw.json', raw)

    print('\n=== T1 summary (FM loss by conditioning, CPU) ===')
    for t, v in sorted(report['by_t'].items(), key=lambda kv: float(kv[0])):
        print(f"t={t}: L_c={v['L_correct']:.4f} dL_wrong={v['dL_wrong']:+.4f} dL_null={v['dL_null']:+.4f}")
    for b, v in report['by_bucket'].items():
        print(f'bucket={b}: dL_wrong={v["dL_wrong"]:+.4f} dL_null={v["dL_null"]:+.4f} '
              f'(n={v["n_correct"]})')


if __name__ == '__main__':
    main()
