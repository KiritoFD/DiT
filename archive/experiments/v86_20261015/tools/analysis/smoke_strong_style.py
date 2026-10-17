"""Real checkpoint/critic gradient smoke only; synthetic latent is NOT an eval."""
import argparse
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.model_io import load_model_from_ckpt
from src.eval.in_mem_eval import _get_cache
from src.train.cli import parse_args
from src.train.strong_style import StrongStyle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    cli = ap.parse_args()
    torch.set_num_threads(4)
    model, model_args = load_model_from_ckpt(cli.ckpt, device='cuda', use_ema=True)
    for name in ('y_callig_embedder', 'y_script_embedder', 'y_char_embedder'):
        module = getattr(model, name, None)
        if module is not None:
            module.requires_grad_(False)
    args = parse_args(['--config', 'src/train/configs/v86_strong_A60.json'])
    if not Path(args.data_csv).exists():
        args.data_csv = 'exp-std/csv/train.csv'
    from types import SimpleNamespace
    hook = StrongStyle(args, 'cuda', SimpleNamespace(info=lambda *args: print(*args)))
    cache = _get_cache('exp-std/csv/eval200_fixed.csv', 187, None, '', model_args)
    g = cache['skels_latent'][:1].repeat(2, 1, 1, 1).cuda()
    x = torch.randn(2, 4, 32, 32, device='cuda')
    kw = dict(y_callig=torch.tensor([0, 1], device='cuda'),
              y_char=torch.zeros(2, dtype=torch.long, device='cuda'),
              y_script=torch.zeros(2, dtype=torch.long, device='cuda'), g=g)
    batch = dict(y_callig_raw=torch.tensor([0, 1]), y_script=torch.zeros(2, dtype=torch.long),
                 std_soft=g.cpu())
    # Artificial thresholds exercise the code; do not export as calibration artifacts.
    hook.targets = torch.full((len(hook.critic.slots),), 20., device='cuda')
    hook.brush_targets = hook.targets
    for mode in ('A', 'B', 'C', 'D'):
        name = {'A': 'A60', 'B': 'B25', 'C': 'C', 'D': 'D'}[mode]
        config_args = parse_args(['--config', 'src/train/configs/v86_strong_' + name + '.json'])
        args.strong_style_ratio = config_args.strong_style_ratio
        args.strong_style_cf_ratio = config_args.strong_style_cf_ratio
        args.strong_style_brush_ratio = config_args.strong_style_brush_ratio
        args.strong_style_mode = mode
        model.train(); model.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            fm = model(x, torch.full((2,), 200., device='cuda'), **kw).float().square().mean()
        aux = hook(model, x, torch.full((2,), .2, device='cuda'), kw, batch, fm, 0)
        aux.backward()
        grads = [p.grad.norm() for p in model.parameters() if p.grad is not None]
        assert torch.isfinite(aux) and sum(grads) > 0 and model.training
        assert all(p.grad is None for p in hook.critic.parameters())
        print('REAL_MODEL_AUX_BACKWARD_OK', mode, float(aux.detach()))


if __name__ == '__main__':
    main()
