"""Fixed-batch EMA gradient replay, NOT historical training gradient records."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def gradient_stats(names, first, second, depth):
    totals = {k: [0., 0., 0.] for k in ('global', 'early', 'middle', 'late', 'other')}
    for name, a, b in zip(names, first, second):
        group = 'other'
        if name.startswith('blocks.'):
            block = int(name.split('.')[1])
            group = ('early', 'middle', 'late')[min(2, block*3//depth)]
        aa = float(a.float().square().sum()) if a is not None else 0.
        bb = float(b.float().square().sum()) if b is not None else 0.
        dot = float((a.float()*b.float()).sum()) if a is not None and b is not None else 0.
        for key in ('global', group):
            totals[key] = [x+y for x, y in zip(totals[key], (aa, bb, dot))]
    return {key: dict(fm_norm=a**.5, writer_norm=b**.5,
                     cosine=dot/(a*b)**.5 if a*b > 0 else None)
            for key, (a, b, dot) in totals.items()}


def main():
    from src.eval.model_io import load_model_from_ckpt
    from src.loss.critic_loss import CriticLoss
    from src.loss.flow_matching import FlowMatching, TIME_SCALE
    from src.train.strong_style import calibrated_hinge, real_mask, writer_margin
    from tools.analysis.diagnose_B25_drift import CKPTS
    torch.set_num_threads(4)
    bank = np.load(ROOT/'scratch/disc_lat_full.npz', allow_pickle=True)
    cfg = json.loads((ROOT/'archive/experiments/v86_20261015/src/train/configs/v86_full_fixed_B25.json').read_text())
    critic = CriticLoss(cfg['critic_ckpt'], 'cuda')
    critic.build_id2slot(cfg['data_csv'], cfg['callig_remap_json'], cfg['font_remap_json'])
    mask = real_mask(critic.slots, 'cuda')
    artifact = json.loads(Path(cfg['strong_style_calibration']).read_text())
    if artifact['slots'] != critic.slots or artifact['critic_sha256'] != hashlib.sha256(Path(cfg['critic_ckpt']).read_bytes()).hexdigest():
        raise ValueError('Calibration identity mismatch')
    targets = torch.tensor([artifact['writer_targets']['0.25'].get(s, 0.) for s in critic.slots], device='cuda')
    valid = [i for i in range(len(bank['z'])) if (str(bank['writer'][i]) in critic.id2callig
             and str(bank['script'][i]) in critic.id2script)]
    indices = np.random.default_rng(0).choice(valid, size=128, replace=False).reshape(4, 32)
    report = dict(protocol='EMA fixed critic-development bank replay, four batches32, clean conditioning; no REPA; not historical training logs or full-batch training estimates',
                  indices=indices.tolist(), checkpoints=CKPTS, stages={})
    out = ROOT/'assets/B25_gradient_replay_20261015.json'
    for stage, checkpoint in CKPTS.items():
        model, args = load_model_from_ckpt(str(ROOT/checkpoint), device='cuda', use_ema=True)
        model.eval()
        for name in ('y_callig_embedder', 'y_script_embedder', 'y_char_embedder'):
            module = getattr(model, name, None)
            if module is not None:
                module.requires_grad_(False)
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        names, parameters = zip(*named)
        diffusion = FlowMatching()
        records = []
        for batch_index, ix in enumerate(indices):
            torch.manual_seed(batch_index)
            torch.cuda.manual_seed_all(batch_index)
            x = torch.tensor(bank['z'][ix], device='cuda').float()
            g = torch.tensor(bank['g_std'][bank['std_idx'][ix].astype(np.int64)], device='cuda').float()[:, None]/255
            writers = [str(bank['writer'][i]) for i in ix]
            scripts = [str(bank['script'][i]) for i in ix]
            kw = dict(y_callig=torch.tensor([critic.id2callig[w] for w in writers], device='cuda'),
                      y_script=torch.tensor([critic.id2script[s] for s in scripts], device='cuda'),
                      y_char=torch.zeros(len(ix), dtype=torch.long, device='cuda'), g=g)
            t = diffusion.sample_t(len(ix), 'cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                fm = diffusion.training_losses(model, x, t, kw)['loss'].mean()
            first = torch.autograd.grad(fm, parameters, allow_unused=True)
            t = torch.rand(len(ix), device='cuda')*.25+.05
            xt = (1-t[:, None, None, None])*x+t[:, None, None, None]*torch.randn_like(x)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                v = model(xt, t*TIME_SCALE, **kw)
            if isinstance(v, tuple):
                v = v[0]
            z = xt-t[:, None, None, None]*v[:, :4].float()
            sid = torch.tensor([critic.s2i[s] for s in scripts], device='cuda')
            yi = torch.tensor([critic.slots.index(w+'·'+s) for w, s in zip(writers, scripts)], device='cuda')
            margin = writer_margin(critic.model(z, g, sid)['slot_logits'], yi, mask)
            writer = calibrated_hinge(margin, targets[yi])
            second = torch.autograd.grad(writer, parameters, allow_unused=True)
            records.append(dict(batch=batch_index, fm=float(fm), writer=float(writer),
                                stats=gradient_stats(names, first, second, len(model.blocks))))
            del first, second
            print(stage, batch_index, records[-1]['stats']['global'], flush=True)
        report['stages'][stage] = records
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        del model, parameters, named
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
