"""Experiment B model C: reference adapter on a FROZEN discrete-ID generator.

Base: control48 pretraining checkpoint (48M, writer_mode=shared, 259 group
rows + trained null row). Everything is frozen except a fresh StyleHead
(768->128) and bridge (128->32) whose output REPLACES the writer embedding
inside the FiLM pathway (writer_slot injection). Training data: Top10 clean
rows with same-group cross-character references (strictly different chars);
the target character is never in its own reference set.

Arms at eval time share this frozen base:
  A  STD+Script            -> writer = null row (trained unconditional path)
  B  STD+WriterID          -> the fine-tuned control48 checkpoint (separate)
  C  STD+cross-char refs   -> this adapter
  C-wrong                  -> same adapter, wrong-writer references (eval swap)
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
from src.loss.flow_matching import FlowMatching
from src.utils.std_soft_glyph import StdSoftGlyphStore

VAE_SCALE = 0.18215
TOP10_RAW = {601: 0, 703: 1, 479: 2, 956: 3, 9001: 4, 760: 5, 672: 6, 365: 7, 461: 8, 53: 9}


def build_c_model(base_ckpt, bank_paths, refs, device='cuda'):
    ck = torch.load(base_ckpt, map_location='cpu', weights_only=False)
    saved = dict(ck.get('config') or ck['args'].get('architecture_config'))
    saved.update(reference_dim=128, reference_injection='writer_slot',
                 reference_runtime='group', reference_cache=str(bank_paths[0]),
                 reference_extra_cache=str(bank_paths[1]) if len(bank_paths) > 1 else '',
                 reference_max_refs=refs, reference_dropout=0.0)
    config = Config(**saved)
    config.validate(require_budget=False)
    model = build(config).to(device)
    weights = {k.removeprefix('_orig_mod.'): v for k, v in ck['ema'].items()}
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if unexpected: raise ValueError(f'unexpected keys {unexpected[:4]}')
    if not all(k.startswith(('reference_head.', 'reference_bridge.')) for k in missing):
        raise ValueError(f'unexpected missing keys {missing[:6]}')
    for p in model.parameters(): p.requires_grad_(False)
    # fresh adapter params are the ONLY trainable tensors
    model.reference_head.requires_grad_(True); model.reference_bridge.requires_grad_(True)
    bank = MergedBank(*bank_paths)
    return model, config, bank, int(ck['train_steps'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='assets/u100_control48_20261024/runs')
    parser.add_argument('--banks', nargs='+', default=['assets/u100_pretrain_20261024/style_bank.npz',
                                                       'assets/u100_pretrain_20261024/style_bank_top10.npz'])
    parser.add_argument('--csv', default='assets/u100_pretrain_20261024/ft_train.csv')
    parser.add_argument('--latents', default='assets/generic48_kxl_top10_20261022/latents')
    parser.add_argument('--std-root', default='assets/generic48_kxl_top10_20261022/std_bbox64')
    parser.add_argument('--tables', default='assets/u100_pretrain_20261024/tables')
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--refs', type=int, default=2)
    parser.add_argument('--out', default='assets/u100_pretrain_20261024/exp_b_c_adapter.pt')
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists(): raise FileExistsError(out)
    torch.manual_seed(90); np.random.seed(90)
    torch.set_num_threads(4)

    runs = sorted(Path(args.base).glob('*/checkpoints/*.pt'))
    base_ckpt = str(runs[-1])
    model, config, bank, base_step = build_c_model(base_ckpt, args.banks, args.refs)
    model.train()
    fremap = json.loads(Path(args.tables, 'font_remap.json').read_text())
    store = StdSoftGlyphStore(args.std_root, sigma_min=.65, sigma_max=.65, eval_sigma=.65)
    process = FlowMatching(num_steps=50, sampler='heun')

    rows = [r for r in csv.DictReader(open(args.csv, encoding='utf-8-sig'))]
    shard_of = {}
    for p in sorted(Path(args.latents).glob('shard_*.npz')):
        d = np.load(p)
        for i, v in enumerate(d['img_ids']): shard_of[int(v)] = (str(p), i)
    missing = [int(r['img_id']) for r in rows if int(r['img_id']) not in shard_of]
    if missing: raise ValueError(f'{len(missing)} rows lack latents, e.g. {missing[:3]}')

    adapter_params = [p for p in model.parameters() if p.requires_grad]
    n_adapter = sum(p.numel() for p in adapter_params)
    opt = torch.optim.AdamW(adapter_params, lr=args.lr, weight_decay=0.01)
    rng = np.random.default_rng(90)
    log_every = 200
    for step in range(args.steps):
        idx = rng.choice(len(rows), args.batch, replace=False)
        batch = [rows[i] for i in idx]
        z0 = torch.stack([torch.from_numpy(
            np.load(shard_of[int(r['img_id'])][0])['latents'][shard_of[int(r['img_id'])][1]].astype(np.float32)
        ) for r in batch]).cuda() * VAE_SCALE
        g = torch.stack([store.get(int(r['script_id']), r['character'], sigma=.65) for r in batch]).cuda()
        script = torch.tensor([fremap[r['script_id']] for r in batch], device='cuda')
        pos = bank.positions([int(r['img_id']) for r in batch])
        refs, mask = bank.sample(pos, np.random.default_rng(1000 + step), args.refs)
        kwargs = dict(reference_features=torch.as_tensor(bank.features[refs]).cuda(),
                      reference_mask=torch.as_tensor(mask).cuda())
        t = torch.rand(len(batch), device='cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            terms = process.training_losses(model, z0, t, model_kwargs=dict(
                g=g, y_callig=torch.zeros(len(batch), dtype=torch.long, device='cuda'),
                y_script=script, **kwargs))
            loss = terms['loss'].mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter_params, 1.0)
        opt.step()
        if step % log_every == 0:
            print(f'ADAPTER step={step} fm={float(loss):.4f} lr={args.lr}', flush=True)
    torch.save(dict(state={k: v.cpu() for k, v in model.state_dict().items()},
                    base_checkpoint=base_ckpt, base_step=base_step,
                    adapter_params=n_adapter, steps=args.steps, lr=args.lr,
                    banks=list(args.banks), refs=args.refs,
                    note='frozen control48 pretrain; trainable StyleHead+bridge only (writer_slot injection)'), out)
    print('ADAPTER_DONE', out, 'adapter_params', n_adapter, flush=True)


if __name__ == '__main__':
    main()
