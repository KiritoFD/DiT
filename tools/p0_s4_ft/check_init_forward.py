"""Gate 2: does the freshly initialised I1 damage content before training starts?

The plan's check #2 asks that under the null Writer the prediction stays basically P0's,
and specifically that the new I1 does not severely change content on step 1.

Why compare velocities instead of rendered samples
--------------------------------------------------
Sampling would be the more end-to-end statement, but it needs the VAE, the STD skeleton
cache, and the full Heun loop -- several minutes of GPU and a lot of surface area for a
pre-flight gate. The quantity I1 actually acts on is the model's velocity output at a
given (x_t, t, writer, script, glyph). If that is close to P0's at the same inputs, the
sampler will produce close outputs, because the sampler only ever consumes this function.

So: build the S4 model with the partial transfer, build P0 in its own architecture, feed
BOTH the identical batch, and compare v. Two probes, because they answer different things:

  * null-writer  -> the plan's check #2. Should be very close; a big gap means I1's random
    init is injecting a strong bias into the content path.
  * named-writer -> expected to be WORSE, because I1 is random and the ten writer rows were
    re-seeded from a mean. If this is as good as the null probe, I1 is doing nothing and
    the arm would not learn a writer condition at all.

Reporting both matters: only the pair distinguishes "I1 is presently harmless" from
"I1 is inert".

The gate's threshold is deliberately loose. This is a smoke check for gross breakage
(NaN, sign flips, an order-of-magnitude change), not a quality metric.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.style_matrix.runtime.common import construct  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
P0_CKPT = ROOT / 'assets/p125_20261030/execution/runs/P0/checkpoints/0050000.pt'
P0_SPEC = ROOT / 'assets/p125_20261030/specs/P0.json'


def build_p0(device):
    """P0 in its own architecture, EMA only, so it is a clean reference."""
    from src.unet.io import Config, build

    spec = json.loads(P0_SPEC.read_text(encoding='utf-8'))
    config = Config(**spec['config'])
    model = build(config).to(device).eval()
    ck = torch.load(P0_CKPT, map_location='cpu', weights_only=False)
    weights = {k.removeprefix('_orig_mod.'): v for k, v in ck['ema'].items()}
    model.load_state_dict(weights, strict=True)
    return model, config


def velocity(model, config, *, batch, device, writer_id, seed=0):
    """One forward at a fixed (x_t, t), identical for both models."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    B = batch
    x0 = torch.randn(B, 4, 32, 32, generator=g)
    noise = torch.randn(B, 4, 32, 32, generator=g)
    t = torch.full((B,), 0.5)
    xt = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * noise
    glyph = torch.rand(B, 1, 64, 64, generator=g) > 0.5
    glyph = glyph.float() * 2 - 1

    y_callig = torch.full((B,), writer_id, dtype=torch.long)
    y_script = torch.zeros(B, dtype=torch.long)
    with torch.no_grad():
        v = model(xt.to(device), (t * 1000.0).to(device), y_callig.to(device),
                  y_script=y_script.to(device), g=glyph.to(device))
    if isinstance(v, (tuple, list)):
        v = v[0]
    return v.float().cpu()


def stats(a, b):
    d = (a - b).float()
    denom = b.abs().mean().clamp_min(1e-8)
    return dict(mean_abs_diff=float(d.abs().mean()),
                rel_to_ref=float(d.abs().mean() / denom),
                cosine=float(torch.nn.functional.cosine_similarity(
                    a.flatten()[None], b.flatten()[None]).item()),
                ref_rms=float(b.square().mean().sqrt()),
                ref_abs_mean=float(denom),
                finite=bool(torch.isfinite(a).all()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', required=True)
    ap.add_argument('--out', default=str(ROOT / 'assets/p0_s4_ft/init_check.json'))
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--null-rel-max', type=float, default=0.5,
                    help='null-writer relative change above this fails the gate')
    ap.add_argument('--cos-min', type=float, default=0.5,
                    help='null-writer cosine below this fails the gate')
    a = ap.parse_args()

    if a.device == 'cuda' and not torch.cuda.is_available():
        a.device = 'cpu'
    print(f'device={a.device} batch={a.batch}')

    spec = json.loads(Path(a.spec).read_text(encoding='utf-8'))
    model, config, transfer = construct(spec, device=a.device)
    model = model.eval()
    print(f'target: arm={spec["arm"]} repr={config.style_matrix_repr} '
          f'injection={config.style_matrix_injection} writers={config.num_writers}')
    print(f'transfer: backbone {transfer["backbone"]["identical"]}/'
          f'{transfer["backbone"]["loaded"]} identical, '
          f'style {transfer["style"]["verified"]} verified, '
          f'dropped {transfer["counts"]["dropped"]} fresh {transfer["counts"]["fresh"]}')

    p0, p0_config = build_p0(a.device)
    print(f'reference: P0 repr={p0_config.style_matrix_repr} '
          f'injection={p0_config.style_matrix_injection} '
          f'writers={p0_config.num_writers}')

    # Writer id 0 exists in both (S4 has 10 named + null; P0 has 47 + null), so the same
    # id is comparable. The null slot is the last row of each table.
    probes = {}
    v_p0_null = velocity(p0, p0_config, batch=a.batch, device=a.device,
                         writer_id=p0_config.num_writers)
    v_s4_null = velocity(model, config, batch=a.batch, device=a.device,
                         writer_id=config.num_writers)
    probes['null_writer'] = stats(v_s4_null, v_p0_null)

    v_p0_w0 = velocity(p0, p0_config, batch=a.batch, device=a.device, writer_id=0)
    v_s4_w0 = velocity(model, config, batch=a.batch, device=a.device, writer_id=0)
    probes['writer_0'] = stats(v_s4_w0, v_p0_w0)

    for name, s in probes.items():
        print(f'\n--- {name} ---')
        print(f'  ref |v| mean      = {s["ref_abs_mean"]:.4f}  (rms {s["ref_rms"]:.4f})')
        print(f'  |dv| mean         = {s["mean_abs_diff"]:.4f}')
        print(f'  relative to ref   = {s["rel_to_ref"]:.2%}')
        print(f'  cosine(v_s4, v_P0)= {s["cosine"]:.4f}')
        print(f'  finite            = {s["finite"]}')

    nulls = probes['null_writer']
    ok = (nulls['finite'] and nulls['rel_to_ref'] <= a.null_rel_max
          and nulls['cosine'] >= a.cos_min)
    verdict = 'PASS' if ok else 'FAIL'

    # An I1 that is indistinguishable from absent would make the arm inert. Report it
    # rather than silently passing: the null probe being good is only meaningful if the
    # named probe shows I1 actually does something.
    named = probes['writer_0']
    inert = abs(named['rel_to_ref'] - nulls['rel_to_ref']) < 1e-6

    print(f'\nverdict: {verdict} (null rel={nulls["rel_to_ref"]:.2%} '
          f'<= {a.null_rel_max:.0%}, cos={nulls["cosine"]:.3f} >= {a.cos_min})')
    if inert:
        print('  NOTE: named-writer change equals null-writer change -- I1 may be inert')

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(spec=a.spec, device=a.device, batch=a.batch,
                                   probes=probes, verdict=verdict, inert=bool(inert),
                                   transfer=dict(backbone=transfer['backbone'],
                                                 style=transfer['style'],
                                                 counts=transfer['counts'])),
                              ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'wrote {out}')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
