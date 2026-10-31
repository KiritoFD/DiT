"""T2 — Writer-injection site ablation (CPU only, inference-time).

Question (plan §T2): which injection scale actually carries the writer, and
where does writer information enter the velocity field?

Method: keep the checkpoint file untouched and zero the 0-d `gain` buffer of
one `style_injections[i]` at a time (`injections.py` multiplies the whole delta
by `gain`, so gain=0 removes exactly that site). Then re-run the FM objective
on the same (z0, noise, t, script, STD) with the correct and a wrong writer.

Readouts per site and per t:
  * dL_site(t)   = L(site off) - L(full)      -> how much that scale matters
  * dL_wrong(t)  = L(wrong) - L(correct)      -> writer discrimination
  * retention    = dL_wrong(off) / dL_wrong(full) -> does the site carry writer?

A site whose removal flattens dL_wrong is the writer channel; a site whose
removal raises loss at low t is content-critical. Spatial size per site is
recorded with a forward hook on the first full pass (the plan's 8×8 / 16×16 /
32×32 grouping comes from this measurement, not from naming assumptions).
"""
import sys
from collections import defaultdict
from pathlib import Path
import statistics as st

import torch

sys.path.insert(0, '.')
from tools.diag.common import (base_args, load_model, read_rows, pin,  # noqa: E402
                               shard_latents, std_store_of, null_writer,
                               pick_rows, font_map, slot_of, write_json)

TS = (0.10, 0.25, 0.40, 0.60, 0.80, 0.95)


def build_diffusion_for(config):
    from src.loss import flow_kwargs_from
    from src.eval.sampler import build_diffusion
    return build_diffusion(999, 'flow', flow_kwargs_from(config))


def site_meta(model):
    """One forward pass with hooks: (kind, channels, spatial) per injection."""
    meta = {}
    handles = []
    for i, inj in enumerate(model.style_injections):
        def hook(mod, args, output, _i=i):
            h = args[0] if args else None
            if torch.is_tensor(h):
                meta[_i] = dict(channels=int(h.shape[1]),
                                spatial=[int(h.shape[-2]), int(h.shape[-1])],
                                kind=int(getattr(mod, 'kind', -1)))
        handles.append(inj.register_forward_hook(hook))
    return meta, handles


def main():
    a = base_args(extra=[('--rows', dict(type=int, default=16))])
    pin(a.threads)
    out_dir = Path(a.out) / 't2'
    out_dir.mkdir(parents=True, exist_ok=True)
    model, cfg, ck = load_model(a.checkpoint, a.device)
    diffusion = build_diffusion_for(cfg)
    null = null_writer(model)
    srow = font_map(cfg)
    sites = model.style_injections
    print(f'[t2] model={a.checkpoint} step={ck["train_steps"]} sites={len(sites)}', flush=True)

    rows = pick_rows(read_rows(a.csv), a.rows)
    img_ids = [int(r['img_id']) for r in rows]
    z0s = shard_latents(a.shards, img_ids, device=a.device)
    store = std_store_of(cfg)

    # --- probe site geometry once (full model, first usable row)
    meta, handles = site_meta(model)
    for r in rows:
        iid = int(r['img_id'])
        if iid not in z0s:
            continue
        try:
            g = store.get(int(r['script_id']), r['character'])
        except Exception:
            continue
        with torch.no_grad():
            gg = torch.as_tensor(g).float()
            gg = gg[None] if gg.dim() == 3 else gg
            gg = gg.to(a.device)
            kw = dict(g=gg,
                      y_callig=torch.tensor([slot_of(r)], device=a.device),
                      y_script=torch.tensor([srow(int(r['script_id']))], device=a.device),
                      condition_dropped=False)
            model(z0s[iid][None], torch.tensor([100.0], device=a.device), **kw)
        break
    for h in handles:
        h.remove()
    print(f'[t2] site geometry: {meta}', flush=True)

    variants = ['full'] + [f'off_{i}' for i in range(len(sites))]
    slots_present = sorted({slot_of(r) for r in rows})
    print(f'[t2] slots present in sample: {slots_present}', flush=True)
    raw = []
    for variant in variants:
        saved = {}
        if variant != 'full':
            idx = int(variant.split('_')[1])
            with torch.no_grad():
                saved[idx] = float(sites[idx].gain.item())
                sites[idx].gain.fill_(0.0)
        try:
            for ri, r in enumerate(rows):
                iid = int(r['img_id'])
                if iid not in z0s:
                    continue
                try:
                    g = store.get(int(r['script_id']), r['character'])
                except Exception:
                    continue
                g = torch.as_tensor(g).float()
                g = g[None] if g.dim() == 3 else g
                g = g.to(a.device)
                slot = slot_of(r)
                others = [s for s in slots_present if s != slot]
                wrong = others[ri % len(others)] if others else (slot + 7) % 47
                s_id = torch.tensor([srow(int(r['script_id']))], device=a.device)
                z0 = z0s[iid][None]
                gen = torch.Generator(device=a.device).manual_seed(iid)
                noise = torch.randn(z0.shape, generator=gen, device=a.device)
                for t in TS:
                    zt = (1.0 - t) * z0 + t * noise
                    tt = torch.full((1,), float(t), device=a.device)
                    for name, w in (('correct', slot), ('wrong', wrong)):
                        kw = dict(g=g, y_callig=torch.tensor([w], device=a.device), y_script=s_id,
                                  condition_dropped=False)
                        with torch.no_grad():
                            out = diffusion.training_losses(model, zt, tt, kw, noise)
                        loss = out['loss'] if isinstance(out, dict) else out
                        raw.append(dict(variant=variant, img=iid, t=t, cond=name,
                                        loss=round(float(loss.mean()), 6)))
        finally:
            for idx, val in saved.items():
                with torch.no_grad():
                    sites[idx].gain.fill_(val)
        print(f'[t2] variant {variant} done ({len(raw)} records)', flush=True)

    if not raw:
        raise SystemExit('[t2] no records — check std-soft coverage / shard ids')

    table = defaultdict(lambda: defaultdict(list))   # (variant,t) -> cond -> [loss]
    for d in raw:
        table[(d['variant'], d['t'])][d['cond']].append(d['loss'])

    def entry(variant, t):
        per = table[(variant, t)]
        lc, lw = st.fmean(per['correct']), st.fmean(per['wrong'])
        return dict(L_correct=round(lc, 6), L_wrong=round(lw, 6),
                    dL_wrong=round(lw - lc, 6), n=len(per['correct']))

    full = {t: entry('full', t) for t in TS}
    report = {'checkpoint': a.checkpoint, 'step': int(ck['train_steps']),
              'site_geometry': {str(k): v for k, v in meta.items()},
              'variants': {}, 'full': {str(t): v for t, v in full.items()}}

    for variant in variants:
        if variant == 'full':
            continue
        per_t, retention = {}, {}
        for t in TS:
            e = entry(variant, t)
            per_t[str(t)] = dict(dL_site=round(e['L_correct'] - full[t]['L_correct'], 6),
                                 dL_wrong=e['dL_wrong'])
            denom = full[t]['dL_wrong']
            # ratio is only meaningful when the baseline actually discriminates
            retention[str(t)] = (round(e['dL_wrong'] / denom, 3)
                                 if abs(denom) >= 5e-3 else None)
        report['variants'][variant] = {'by_t': per_t,
                                       'writer_retention': retention,
                                       'mean_dL_site': round(
                                           st.fmean(v['dL_site'] for v in per_t.values()), 6),
                                       'mean_retention': (round(st.fmean(
                                           [v for v in retention.values() if v is not None]), 3)
                                           if any(v is not None for v in retention.values())
                                           else None)}

    ranked = sorted(((v, report['variants'][v]['mean_dL_site'])
                     for v in report['variants']), key=lambda kv: -abs(kv[1]))
    report['verdict_hint'] = {
        'content_critical_site': ranked[0][0] if ranked else None,
        'note': 'dL_site>0 means the site was helping; retention near 0 means the '
                'site carried writer discrimination',
    }
    write_json(out_dir / 't2_report.json', report)

    print('\n=== T2 summary (FM loss change when one injection site is disabled) ===')
    for k, v in report['site_geometry'].items():
        print(f'site {k}: kind={v["kind"]} ch={v["channels"]} spatial={v["spatial"]}')
    for variant, v in report['variants'].items():
        print(f'{variant}: mean_dL_site={v["mean_dL_site"]:+.4f} '
              f'mean_writer_retention={v["mean_retention"]}')


if __name__ == '__main__':
    main()
