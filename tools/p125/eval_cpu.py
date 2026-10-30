"""CPU evaluation of P125 checkpoints while training owns the GPU.

Matrix eval kernel (src/eval/in_mem_eval.py) threads every tensor through an
explicit device argument, so loading on CPU covers sampling/decode/metrics.
Infra rules copied from src/eval/cpu_eval_worker.py (the DiT-line CPU eval
daemon): pin torch thread pools, small atomic outputs, one process per eval.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import torch
from src.unet.io import load, evaluate
from tools.style_matrix.runtime.common import save


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--csv', action='append', required=True,
                   help='repeatable: each gets an out/<stem> directory')
    p.add_argument('--out', required=True)
    p.add_argument('--threads', type=int, default=48)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    try:
        torch.set_num_interop_threads(2)
    except RuntimeError:
        pass
    torch.set_grad_enabled(False)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model, cfg, ck = load(a.checkpoint, 'cpu', use_ema=True)
    step = int(ck['train_steps'])
    for csv_path in a.csv:
        name = Path(csv_path).stem
        run_dir = out / f'{name}_{step:07d}'
        run_dir.mkdir(parents=True, exist_ok=True)
        callig = str(run_dir / 'writer_rows.json')
        save(callig, {str(i): i for i in range(cfg.num_writers)})
        cfg_run = replace(cfg, eval_csv=csv_path, callig_remap_json=callig,
                          output=str(run_dir))
        print(f'[eval-cpu] threads={a.threads} step={step} csv={csv_path}', flush=True)
        result = evaluate(model, cfg_run, step, str(run_dir), use_ema=True)
        save(run_dir / 'result.json', result.get('eval200fix', result))
        m = result.get('eval200fix', result)
        print('[eval-cpu]', name, json.dumps(
            {k: m.get(k) for k in ('ssim', 'mse', 'lpips', 'ink_iou', 'skel_iou', 'frag', 'n')}),
            flush=True)


if __name__ == '__main__':
    main()
