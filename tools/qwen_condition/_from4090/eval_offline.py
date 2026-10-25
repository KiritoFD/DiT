"""Offline held-out evaluations for a trained U100 checkpoint.

Rewrites calligrapher IDs to the frozen Top10 rows, then evaluates the named
sets with explicit STD sigma per set. Used after pretraining/FT for eval200fix
(U100-protocol comparison), test200 and val100 at the fine-tune sigma.
"""
import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.unet.io import load, evaluate, checkpoint_config, build, eval_args
import torch

TOP10_RAW = {601: 0, 703: 1, 479: 2, 956: 3, 9001: 4, 760: 5, 672: 6, 365: 7, 461: 8, 53: 9}


def load_legacy_ema_verified(path):
    """Audited exception for pre-architecture checkpoints: strict EMA load plus
    verification that EMA actually tracked training (differs from raw, finite)."""
    ck = torch.load(path, map_location='cpu', weights_only=False)
    config = checkpoint_config(ck)
    model = build(config).to('cuda')
    weights = {k.removeprefix('_orig_mod.'): v for k, v in ck['ema'].items()}
    model.load_state_dict(weights, strict=True)
    raw = {k.removeprefix('_orig_mod.'): v for k, v in ck['delta'].items()}
    differing = sum(1 for k in weights if not torch.equal(weights[k], raw[k].to(weights[k].dtype)))
    if differing < len(weights) * 0.5:
        raise ValueError(f'Legacy EMA looks untracked: only {differing}/{len(weights)} tensors differ from raw')
    if not all(torch.isfinite(v).all() for v in weights.values()):
        raise ValueError('Legacy EMA contains non-finite values')
    model.eval()
    record = dict(differing_tensors=differing, total_tensors=len(weights),
                  note='legacy EMA verified: strict load, majority-differ from raw, finite')
    print('LEGACY_EMA_VERIFIED', record, flush=True)
    return model, config, ck, record


def rewrite(src, dst):
    rows = list(csv.DictReader(Path(src).open(encoding='utf-8-sig')))
    for r in rows:
        raw = int(r['calligrapher_id'])
        if raw not in TOP10_RAW: raise ValueError(f'writer {raw} outside Top10 in {src}')
        r['calligrapher_id'] = str(TOP10_RAW[raw])
    with Path(dst).open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    return dst


def extend_char_remap(config, sets, out):
    """Held-out protocol sets can carry character IDs outside the pool mapping;
    character IDs never enter the model, so appending dense ids is identity-safe."""
    charmap = json.loads(Path(config.char_remap_json).read_text(encoding='utf-8'))
    known = {int(k) for k in charmap}
    seen = set()
    for _, csv_path, _ in sets:
        for r in csv.DictReader(Path(csv_path).open(encoding='utf-8-sig')):
            seen.add(int(r['character_id']))
    missing = sorted(seen - known)
    if not missing: return config.char_remap_json
    nxt = max(known) + 1
    for cid in missing:
        charmap[str(cid)] = nxt; nxt += 1
    target = out / 'sets' / 'char_remap_extended.json'
    target.write_text(json.dumps(charmap), encoding='utf-8')
    return str(target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--eval200fix', default='exp-std/csv/eval200_fixed.csv')
    parser.add_argument('--test200', default='assets/generic48_kxl_top10_20261022/top10_test.csv')
    parser.add_argument('--val100', default='')
    parser.add_argument('--sets', default='eval200fix,test200',
                        help='comma list from: eval200fix,test200,val100')
    parser.add_argument('--identity-callig', action='store_true',
                        help='CSV calligrapher ids are already 0..9; use an identity remap '
                             'instead of the checkpoint-origin raw-id remap')
    parser.add_argument('--allow-legacy-ema', action='store_true',
                        help='Verified EMA load for pre-architecture checkpoints')
    args = parser.parse_args()
    wanted = [s.strip() for s in args.sets.split(',') if s.strip()]
    out = Path(args.out); (out / 'sets').mkdir(parents=True, exist_ok=True)
    all_sets = [('eval200fix', rewrite(args.eval200fix, out / 'sets' / 'eval200fix.csv'), 0.65),
                ('test200', rewrite(args.test200, out / 'sets' / 'test200.csv'), 0.0)]
    if args.val100: all_sets.append(('val100', args.val100, 0.0))
    sets = [entry for entry in all_sets if entry[0] in wanted]
    legacy_record = None
    if args.allow_legacy_ema:
        model, config, ck, legacy_record = load_legacy_ema_verified(args.checkpoint)
    else:
        model, config, ck = load(args.checkpoint, 'cuda', use_ema=True)
    if args.identity_callig:
        identity = out / 'sets' / 'callig_remap_identity.json'
        identity.write_text(json.dumps({str(i): i for i in range(config.num_writers)}), encoding='utf-8')
        config = replace(config, callig_remap_json=str(identity))
    char_remap = extend_char_remap(config, sets, out)
    results = {}
    for name, csv_path, sigma in sets:
        cfg = replace(config, eval_csv=str(csv_path), std_soft_eval_sigma=sigma,
                      char_remap_json=char_remap)
        with torch.inference_mode():
            metrics = evaluate(model, cfg, int(ck['train_steps']), out / name)
        results[name] = {k: (float(v) if isinstance(v, (int, float)) else v)
                         for k, v in metrics.items() if isinstance(v, (int, float))}
        print('EVAL_DONE', name, json.dumps(results[name]), flush=True)
    (out / 'offline_eval.json').write_text(json.dumps(
        dict(checkpoint=args.checkpoint, step=int(ck['train_steps']),
             legacy_ema=legacy_record, results=results), indent=2))


if __name__ == '__main__':
    main()
