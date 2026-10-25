"""Measure FM vs SupCon gradient-norm ratios to choose the SupCon weight.

Runs a handful of real batches (pool latents + bank references + STD store)
through the writer_slot model and reports, per parameter group (style head,
bridge, writer_proj, one decoder FiLM), the median ||grad L_FM|| and
||grad L_supcon||. The recommended lambda targets a meaningful but
sub-dominant SupCon share on the head (~0.3x FM) and a bounded trunk share.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.unet.io import Config, build
from src.unet.reference.group_runtime import MergedBank
from src.unet.reference.encoder import supcon
from src.loss.flow_matching import FlowMatching
from src.utils.std_soft_glyph import StdSoftGlyphStore

VAE_SCALE = 0.18215


def group_params(model):
    head = [p for n, p in model.named_parameters() if n.startswith('reference_head.')]
    bridge = [p for n, p in model.named_parameters() if n.startswith('reference_bridge.')]
    proj = [p for n, p in model.named_parameters() if n.startswith('writer_proj.')]
    film = [p for n, p in model.named_parameters() if n.startswith('decoder.0.blocks.0.film.')]
    return dict(head=head, bridge=bridge, writer_proj=proj, decoder_film0=film)


def grad_norm(loss, params):
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    total = 0.
    for g in grads:
        if g is not None: total += float(g.norm() ** 2)
    return total ** .5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='assets/u100_pretrain_20261024/train_100k.csv')
    parser.add_argument('--bank', default='assets/u100_pretrain_20261024/style_bank.npz')
    parser.add_argument('--latents', default='assets/generic48_kxl_top10_20261022/latents')
    parser.add_argument('--std-root', default='assets/generic48_kxl_top10_20261022/std_bbox64')
    parser.add_argument('--tables', default='assets/u100_pretrain_20261024/tables')
    parser.add_argument('--steps', type=int, default=12)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--out', default='assets/u100_pretrain_20261024/lambda_calibration.json')
    args = parser.parse_args()
    torch.manual_seed(90)
    config = Config(channels=(96, 192, 256), num_res_blocks=8, num_writers=259, num_scripts=3,
                    writer_mode='none', decouple_writer_condition=False, reference_dim=128,
                    reference_injection='writer_slot', reference_dropout=.25,
                    reference_cache=args.bank, condition_tables_root=args.tables,
                    batch=args.batch, max_steps=1, stop_at_step=1, freeze_writer_table=True)
    config.validate(require_budget=False)
    model = build(config).cuda().train()
    bank = MergedBank(args.bank)
    store = StdSoftGlyphStore(args.std_root, sigma_min=.65, sigma_max=.65, eval_sigma=.65)
    fremap = json.loads(Path(args.tables, 'font_remap.json').read_text())
    rows = list(csv.DictReader(open(args.csv, encoding='utf-8-sig')))
    lat = {}
    for p in sorted(Path(args.latents).glob('shard_*.npz')):
        d = np.load(p)
        for i, v in enumerate(d['img_ids']): lat[int(v)] = (p, i)
    params = group_params(model)
    rng = np.random.default_rng(90)
    ratios = {k: [] for k in params}
    for step in range(args.steps):
        idx = rng.choice(len(rows), args.batch, replace=False)
        batch = [rows[i] for i in idx]
        z0 = torch.stack([torch.from_numpy(np.load(lat[int(r['img_id'])][0])['latents'][lat[int(r['img_id'])][1]].astype(np.float32)) for r in batch]).cuda() * VAE_SCALE
        g = torch.stack([store.get(int(r['script_id']), r['character'], sigma=.65) for r in batch]).cuda()
        script = torch.tensor([fremap[r['script_id']] for r in batch], device='cuda')
        pos = bank.positions([int(r['img_id']) for r in batch])
        refs, mask = bank.sample(pos, np.random.default_rng(step), 2)
        ref_kwargs = dict(reference_features=torch.as_tensor(bank.features[refs]).cuda(),
                          reference_mask=torch.as_tensor(mask).cuda())
        process = FlowMatching(num_steps=50, sampler='heun')
        t = torch.rand(len(batch), device='cuda')

        def fm_loss():
            terms = process.training_losses(model, z0, t, model_kwargs=dict(
                g=g, y_callig=torch.zeros(len(batch), dtype=torch.long, device='cuda'),
                y_script=script, **ref_kwargs))
            return terms['loss'].mean()
        def sc_loss():
            target = bank.features[pos]
            feat = model.reference_head(torch.as_tensor(target).cuda())
            return supcon(feat, *[torch.as_tensor(v).cuda() for v in
                                  (bank.writers[pos], bank.chars[pos], bank.scripts[pos])])[0]
        lf, ls = fm_loss(), sc_loss()
        for name, ps in params.items():
            if not ps: continue
            gf, gs = grad_norm(lf, ps), grad_norm(ls, ps)
            if gf > 0 and gs > 0: ratios[name].append(gs / gf)
    report = {k: dict(median_ratio=float(np.median(v)), n=len(v)) for k, v in ratios.items() if v}
    head_ratio = report.get('head', {}).get('median_ratio', 1.0)
    report['recommended_lambda'] = round(0.3 / head_ratio, 4)
    report['rule'] = 'lambda * gradSup(head) ~= 0.3 * gradFM(head)'
    Path(args.out).write_text(json.dumps(report, indent=2))
    print('LAMBDA_CALIBRATION', json.dumps(report))


if __name__ == '__main__':
    main()
