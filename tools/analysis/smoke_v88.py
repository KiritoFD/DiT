"""GPU preflight: native shape, full batch, optimizer, gradients, strict eval reload."""
import json
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.train.cli import parse_args
from src.model.build import build_dit_from_args
from src.loss.flow_matching import FlowMatching
from src.train.writer_table_tuning import configure_writer_table, split_writer_optimizer_group


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='src/train/configs/v88_writer_glyph_base.json')
    parser.add_argument('--out', default='assets/v88_preflight')
    options = parser.parse_args()
    output = Path(options.out)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    args = parse_args(['--config', options.config])
    assert not args.warm_start_from and not args.resume_full
    assert args.strong_style_mode == 'off'
    if not Path(args.callig_table_anchor_file).is_file():
        raise FileNotFoundError('Missing writer anchor; prepare it explicitly before launching. Do not overwrite a reviewed config to recreate an anchor.')
    model = build_dit_from_args(args, device='cuda')
    for name, file in [('y_callig_embedder', 'callig_table.npy'), ('y_script_embedder', 'font_table.npy')]:
        module = getattr(model, name)
        data = torch.from_numpy(np.load(Path(args.triple_table_prefix)/file)).float().cuda()
        with torch.no_grad():
            module.embedding_table.weight[:len(data)].copy_(data)
        module.requires_grad_(False)
    tuning = configure_writer_table(model, args)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    split_writer_optimizer_group(optimizer, tuning, args.callig_table_lr)
    assert [group['lr'] for group in optimizer.param_groups] == [args.lr, args.callig_table_lr]
    model.train()
    # Match production's existing compiler/fusion path; this is not recomputation.
    original = model
    if args.compile:
        model = torch.compile(model, mode=args.compile_mode)
    b = args.global_batch_size
    x = torch.randn(b, 4, 32, 32, device='cuda')
    glyph = torch.rand(b, 1, 64, 64, device='cuda')
    kw = dict(y_callig=torch.arange(b, device='cuda') % 10,
              y_script=torch.arange(b, device='cuda') % 3,
              y_char=torch.zeros(b, dtype=torch.long, device='cuda'), g=glyph)
    diffusion = FlowMatching()
    start = time.monotonic()
    history = []
    for step in range(8):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = diffusion.training_losses(model, x, diffusion.sample_t(b, 'cuda'), kw)['loss'].mean()
        loss.backward()
        assert torch.isfinite(loss)
        branch = model.writer_glyph
        internal_norm = sum(float(p.grad.float().norm()) for p in branch.film.P.parameters() if p.grad is not None)
        history.append(dict(step=step, loss=float(loss), gains={k: float(v) for k, v in branch.gains.items()},
                            p_grad_norm=internal_norm))
        optimizer.step()
        tuning.restore_null()
    assert history[-1]['p_grad_norm'] > 0, history
    checkpoint = output/'checkpoint.pt'
    torch.save({'model': original.state_dict(), 'args': args}, checkpoint)
    from src.eval.model_io import load_model_from_ckpt
    restored, _ = load_model_from_ckpt(str(checkpoint), device='cuda', use_ema=False)
    assert restored.writer_glyph.at == tuple(int(i) for i in args.writer_glyph_at.split(','))
    report = dict(batch=b, steps=len(history), elapsed=time.monotonic()-start,
                  peak_gib=torch.cuda.max_memory_allocated()/2**30,
                  strict_eval_reload=True, config=options.config, history=history)
    (output/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
