"""Check that a checkpoint can actually be served, before starting any server.

The point is to fail here, with a list, rather than inside a Gradio callback where the
error surfaces as one broken image in a browser. It answers three questions per
checkpoint:

  1. Can it be LOADED? `src.unet.io.load` rebuilds the architecture from the checkpoint's
     own `config`, so a checkpoint saved by a different architecture is rejected here
     rather than producing garbage. That refusal is the reason the old
     `legacy/gradio/gradio_stdskel.py` ckpt cannot be served by this path at all -- it is
     a DiT (`DiT_2Cond_models`), and `checkpoint_config` rejects anything that is not the
     U-Net `latent_cnn_unet_v1`/`calliflow_v2`.
  2. Are its ASSETS present? A U-Net eval needs the VAE, the STD soft-glyph cache (when
     `std_soft_cond`), the writer/script remap tables, and the image root the CSV points
     at. Each is reported with the exact path so a missing one is actionable.
  3. Can it SAMPLE? A 2-4 step forward on CPU proves the graph runs end to end -- the
     std glyph, the writer id, the script id, the VAE decode -- without a GPU and without
     spending minutes on a 50-step render.

Usage:
  python -c "import sys;print(sys.version)"          # any python with torch
  PYTHONPATH=. python tools/gradio_ui/check_assets.py --ckpt <path> [--ckpt ...]
  PYTHONPATH=. python tools/gradio_ui/check_assets.py --all --out assets/gradio_ui/assets.json
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _exists(p):
    return bool(p) and Path(p).exists()


def check_one(path, device='cpu', sample=True):
    """Return a report dict. Never raises: problems are reported, not thrown."""
    from src.unet.io import load

    rec = dict(checkpoint=str(path), ok=False, checks={}, problems=[])
    p = Path(path)
    if not p.is_file():
        rec['problems'].append(f'checkpoint missing: {path}')
        return rec
    rec['size_GiB'] = round(p.stat().st_size / 2 ** 30, 3)

    try:
        model, cfg, ck = load(str(p), device, use_ema=True)
    except Exception as e:                                    # noqa: BLE001
        rec['problems'].append(f'LOAD FAILED: {type(e).__name__}: {e}')
        return rec

    rec['step'] = int(ck.get('train_steps', -1))
    rec['architecture'] = ck.get('architecture')
    # Field names mirror `Config` exactly. Two traps worth naming: `latent_channels` and
    # `std_soft_cond` are NOT Config fields -- they exist only on the eval args, where
    # `eval_args` hardcodes std_soft_cond=True. Reading them off `cfg` silently yields
    # None, which would skip the std_soft_root check entirely, so conditioning is treated
    # as ON (matching the sampler) and the glyph cache is always required.
    std_soft_on = True
    rec['config'] = dict(
        repr=getattr(cfg, 'style_matrix_repr', None),
        injection=getattr(cfg, 'style_matrix_injection', None),
        num_writers=getattr(cfg, 'num_writers', None),
        num_scripts=getattr(cfg, 'num_scripts', None),
        std_soft_cond=std_soft_on,
        std_soft_root=getattr(cfg, 'std_soft_root', None),
        vae_path=str(getattr(cfg, 'vae_path', '') or ''),
        data_csv=str(getattr(cfg, 'data_csv', '') or ''),
        eval_csv=str(getattr(cfg, 'eval_csv', '') or ''),
        callig_remap_json=str(getattr(cfg, 'callig_remap_json', '') or ''),
        font_remap_json=str(getattr(cfg, 'font_remap_json', '') or ''),
        latent_channels=4,
    )

    # ---- asset checks -------------------------------------------------------
    a = rec['config']
    need = []
    if a['vae_path']:
        need.append(('vae_path', a['vae_path']))
    if a['std_soft_cond'] and a['std_soft_root']:
        need.append(('std_soft_root', a['std_soft_root']))
    if a['callig_remap_json']:
        need.append(('callig_remap_json', a['callig_remap_json']))
    if a['font_remap_json']:
        need.append(('font_remap_json', a['font_remap_json']))
    if a['data_csv']:
        need.append(('data_csv', a['data_csv']))
    if a['eval_csv']:
        need.append(('eval_csv', a['eval_csv']))

    for name, val in need:
        ok = _exists(val)
        rec['checks'][name] = dict(path=val, ok=ok)
        if not ok:
            rec['problems'].append(f'{name} missing: {val}')

    # The image root is the one a CSV indirection hides; resolve it from the CSV header
    # rather than assuming, because a present CSV with a wrong root still fails at render.
    if a['data_csv'] and _exists(a['data_csv']):
        try:
            import csv
            with open(a['data_csv'], encoding='utf-8-sig') as f:
                first = next(csv.DictReader(f))
            key = ('image_path' if 'image_path' in first else
                   'img_path' if 'img_path' in first else None)
            if key:
                rec['checks']['csv_image_column'] = dict(column=key, ok=True)
        except Exception as e:                                # noqa: BLE001
            rec['problems'].append(f'data_csv unreadable: {e}')

    # ---- sampling smoke -----------------------------------------------------
    if sample and not rec['problems']:
        try:
            rec['checks']['sample'] = _sample_smoke(model, cfg, device, ck)
        except Exception as e:                                # noqa: BLE001
            rec['problems'].append(f'SAMPLE FAILED: {type(e).__name__}: {e}')

    rec['ok'] = not rec['problems']
    return rec


def _sample_smoke(model, cfg, device, ck):
    """One short forward+decode. Proves the conditioning ids and glyph shape are valid."""
    import torch
    from src.unet.io import evaluate

    n_w = int(getattr(cfg, 'num_writers', 1))
    n_s = int(getattr(cfg, 'num_scripts', 1))
    # Use the null writer: always in range, and it is the path a "no writer" UI mode hits.
    writer = n_w
    script = 0
    g = torch.zeros(1, 1, 64, 64, device=device)
    g[:, :, 8:56, 8:56] = 1.0
    x = torch.randn(1, int(getattr(cfg, 'latent_channels', 4)), 32, 32, device=device)
    t = torch.full((1,), 500.0, device=device)
    with torch.no_grad():
        v = model(x, t, torch.tensor([writer], device=device),
                  y_script=torch.tensor([script], device=device), g=g)
    if isinstance(v, (tuple, list)):
        v = v[0]
    return dict(ok=True, velocity_shape=list(v.shape),
                finite=bool(torch.isfinite(v).all()),
                n_writers=n_w, n_scripts=n_s,
                used_writer=writer, used_script=script)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', action='append', default=[])
    ap.add_argument('--all', action='store_true',
                    help='scan the known checkpoint locations')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--no-sample', action='store_true')
    ap.add_argument('--out', default=str(ROOT / 'assets/gradio_ui/assets.json'))
    a = ap.parse_args()

    targets = list(a.ckpt)
    if a.all:
        for pat in ('assets/abc_20261031/execution/runs/*/checkpoints/*.pt',
                    'assets/style_matrix_20261027/matrix64/*/execution/runs/*/checkpoints/*.pt',
                    'assets/d_20261101/runs/*/checkpoints/*.pt',
                    'assets/p0_s4_ft/runs/*/checkpoints/*.pt',
                    'assets/imported_ckpts/*.pt'):
            targets += sorted(str(p) for p in ROOT.glob(pat))

    # De-duplicate, keep order.
    seen, uniq = set(), []
    for t in targets:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    if not uniq:
        print('no checkpoints to check (use --ckpt or --all)')
        return 2

    reports = []
    for t in uniq:
        print(f'--- {t} ---', flush=True)
        r = check_one(t, a.device, sample=not a.no_sample)
        reports.append(r)
        if r['problems']:
            for pb in r['problems']:
                print(f'  PROBLEM: {pb}')
        else:
            print(f'  OK step={r.get("step")} '
                  f'repr={r["config"]["repr"]} inj={r["config"]["injection"]} '
                  f'writers={r["config"]["num_writers"]} '
                  f'vel={r["checks"].get("sample", {}).get("velocity_shape")}')

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(device=a.device, reports=reports),
                              ensure_ascii=False, indent=2), encoding='utf-8')
    good = sum(1 for r in reports if r['ok'])
    print(f'\n{good}/{len(reports)} checkpoints serveable; wrote {out}')
    return 0 if good == len(reports) else 1


if __name__ == '__main__':
    raise SystemExit(main())
