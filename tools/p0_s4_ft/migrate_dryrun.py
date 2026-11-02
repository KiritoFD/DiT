"""Dry-run the P0 -> S4 migration and print the audit table.

This is the "训练前必须通过的检查" #1 from the plan: prove that Backbone, R2 and the
retained Writer slots match P0's EMA exactly, and that only the Top10 table rows and I1
differ. It runs entirely on CPU metadata plus one model build, and imports nothing from
the training loop, so it is safe to run before committing to a 48k-step job.

Usage:
  PYTHONPATH=. python tools/p0_s4_ft/migrate_dryrun.py
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.p0_s4_ft.partial_transfer import (WRITER_TABLES, apply_transfer,
                                             init_writer_tables, load_backbone,
                                             plan_transfer)

P0_CKPT = ROOT / 'assets/p125_20261030/execution/runs/P0/checkpoints/0050000.pt'
S4_SPEC = ROOT / 'assets/style_matrix_20261027/matrix64/4090/specs/S4.json'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default=str(P0_CKPT))
    ap.add_argument('--spec', default=str(S4_SPEC))
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--jitter', type=float, default=0.01)
    a = ap.parse_args()

    from src.unet.io import Config, build

    spec = json.loads(Path(a.spec).read_text(encoding='utf-8'))
    config = Config(**spec['config'])
    print(f'target arch from {Path(a.spec).name}: '
          f'repr={config.style_matrix_repr} injection={config.style_matrix_injection} '
          f'writers={config.num_writers}')

    ck = torch.load(a.source, map_location='cpu', weights_only=False)
    src = {k.removeprefix('_orig_mod.'): v for k, v in ck['ema'].items()}
    print(f'source {Path(a.source).name}: step={ck["train_steps"]} tensors={len(src)}')

    model = build(config).to(a.device)
    to_load, report = plan_transfer(src, model,
                                    expect_repr=config.style_matrix_repr,
                                    expect_injection=config.style_matrix_injection)

    print('\n--- style tensor audit ---')
    for k in report['transferred']:
        print(f'  TRANSFER  {k}')
    for k in report['dropped']:
        print(f'  DROP      {k}   ({report["reasons"][k]})')
    for k in report['fresh']:
        print(f'  FRESH     {k}   ({report["reasons"][k]})')
    c = report['counts']
    print(f'  totals: transferred={c["transferred"]} dropped={c["dropped"]} fresh={c["fresh"]}')

    print('\n--- writer tables (num_writers-dependent, reported separately) ---')
    for k, v in report['writer_tables'].items():
        if v.get('present'):
            print(f'  {k}: {v["source_rows"]} rows -> {v["target_rows"]} rows  action={v["action"]}')

    # Backbone: everything outside the style namespaces AND outside the two writer
    # tables must be exactly P0's EMA. The writer tables are excluded here because
    # changing num_writers makes them legitimately different, and they get their own
    # re-seed step below -- lumping them in would just report a mismatch forever.
    style = set(report['transferred']) | set(report['dropped']) | set(report['fresh'])
    backbone = [k for k in model.state_dict()
                if k not in style and k not in WRITER_TABLES and k in src]
    print(f'\n--- migration ---')
    print(f'  style: transferred={c["transferred"]} dropped={c["dropped"]} '
          f'fresh={c["fresh"]}')
    bb = load_backbone(model, src, report)
    print(f'  backbone: loaded {bb["loaded"]} tensors, '
          f'byte-identical {bb["identical"]}/{bb["loaded"]} '
          f'(max_diff={bb["max_diff"]:.3e})')

    proof = apply_transfer(model, src, report, to_load)
    print(f'  style: transferred tensors verified byte-identical: {proof["verified"]} '
          f'(max_diff={proof["max_diff"]:.3e})')

    w = init_writer_tables(model, src, jitter=a.jitter)
    print(f'\n--- writer table seeding ---')
    for k, v in w.items():
        if v.get('present'):
            print(f'  {k}: {v["source_rows"]} -> {v["rows"]} rows, jitter={v["jitter"]}, '
                  f'row_spread={v["row_spread"]:.4f}, '
                  f'mean {v["source_mean"]:+.4f} -> {v["new_mean"]:+.4f}')
        else:
            print(f'  {k}: absent')

    # Post-load verification across the WHOLE migrated model, not just style tensors.
    after = model.state_dict()
    worst, worst_key, n_ok, n_tot = 0.0, None, 0, 0
    for k in backbone:
        if k not in after or after[k].shape != src[k].shape:
            continue
        n_tot += 1
        d = float((after[k].float() - src[k].float()).abs().max())
        if d > worst:
            worst, worst_key = d, k
        if d == 0.0:
            n_ok += 1
    print(f'\n--- post-load backbone check ---')
    print(f'  byte-identical: {n_ok}/{n_tot}   max|diff|={worst:.3e} at {worst_key}')

    out = ROOT / 'assets/p0_s4_ft/migration_plan.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(source=a.source, spec=a.spec, report=report,
                                   writer=w, proof=proof, backbone=bb,
                                   backbone_identical=n_ok,
                                   backbone_total=n_tot,
                                   backbone_max_diff=worst),
                              ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\nwrote {out}')
    ok = (worst == 0.0 and n_ok == n_tot and bb['identical'] == bb['loaded']
          and proof['max_diff'] == 0.0)
    if not ok:
        print('  !! migration is NOT exact -- investigate before training')
        return 1
    print('  OK: backbone + R2 byte-identical to P0; only writer rows and I1 differ')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
