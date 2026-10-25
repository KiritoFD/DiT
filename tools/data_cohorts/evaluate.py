"""Ten triangular held-out tests, strict 30k EMA and per-image paired noise."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from .train_serial import save, stamp


ROOT = Path('assets/data_cohorts/top20_20261021_v1')
TRAIN = Path('assets/results/clean_cohorts_20261021/launch_30k_lr7e5')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def checkpoint(size):
    paths = list((TRAIN/f'top{size}_u114').glob('*/checkpoints/0030000.pt'))
    if len(paths) > 1:
        raise ValueError(f'Ambiguous Top{size} checkpoints')
    return paths[0] if paths else None


def worker(args):
    import torch
    from src.unet.io import load, eval_args
    from src.eval.in_mem_eval import run_in_mem_eval
    torch.set_num_threads(4)
    torch.manual_seed(90)
    torch.cuda.manual_seed_all(90)
    ckpt = checkpoint(args.model)
    model, config, saved = load(str(ckpt), 'cuda', use_ema=True)
    assert saved['train_steps'] == 30000
    model.to(memory_format=torch.channels_last)
    test = ROOT/f'datasets_final/top{args.test}/test.csv'
    items = rows(test)
    remap = json.loads(Path(config.callig_remap_json).read_text())
    assert all(str(r['calligrapher_id']) in remap for r in items)
    assert all(r['eval_noise_seed'] for r in items)
    config.eval_csv = str(test)
    config.eval_steps = 50
    ev = eval_args(config)
    ev.in_mem_eval_sets = f'test_top{args.test}:{test}:{len(items)}'
    ev.in_mem_eval_batch = 4
    ev.in_mem_eval_vae_batch = 1
    ev.in_mem_eval_poster = False  # retain every generated PNG; avoid enormous auto posters
    ev.eval_use_ema = True
    out = Path(args.output)/f'model{args.model}_test{args.test}'
    out.mkdir(parents=True, exist_ok=True)
    save(out/'identity.json', dict(model=args.model,test=args.test,step=30000,ema=True,
        checkpoint=str(ckpt),checkpoint_sha256=sha(ckpt),test_csv=str(test),test_sha256=sha(test),
        cfg=1.,sampler='heun',steps=50,noise='per-image eval_noise_seed',n=len(items)))
    with torch.inference_mode():
        # Use the common public signature; remote stable evaluator has no override kwarg.
        run_in_mem_eval(model, ev, 30000, 'cuda', results_dir=str(out))
    metrics = rows(out/'eval_stdskel_summary.csv')[-1]
    assert int(metrics['n']) == len(items)
    assert metrics['lpips_mean'] and metrics['ink_iou_mean']
    save(out/'RESULT.json', dict(model=args.model,test=args.test,**metrics))


def queue(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with (out/'started.lock').open('x') as f:
        f.write(stamp())
    completed = []
    state = dict(started=stamp(),state='running',completed=completed)
    try:
        for model in (20,10,5,2):
            while checkpoint(model) is None:
                status=json.loads((ROOT/'launch_30k_lr7e5/status.json').read_text())
                if status['state'] in ('failed','superseded'):
                    raise RuntimeError(f'Training queue {status["state"]}')
                state.update(state='waiting_checkpoint',model=model,updated=stamp())
                save(out/'status.json',state)
                time.sleep(60)
            # Async checkpoint writers drain before queue marks this cohort completed.
            while model not in json.loads((ROOT/'launch_30k_lr7e5/status.json').read_text())['completed']:
                time.sleep(15)
            for test in (20,10,5,2):
                if test > model:
                    continue
                state.update(state='evaluating',model=model,test=test,updated=stamp())
                save(out/'status.json',state)
                with (out/f'model{model}_test{test}.log').open('xb') as log:
                    subprocess.run([sys.executable,'-u','-m','tools.data_cohorts.evaluate',
                        '--worker','--model',str(model),'--test',str(test),'--output',str(out)],
                        stdout=log,stderr=subprocess.STDOUT,check=True)
                completed.append(json.loads((out/f'model{model}_test{test}/RESULT.json').read_text()))
                save(out/'results.json',completed)
                with (out/'results.csv').open('w',encoding='utf-8-sig',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(completed[0]));w.writeheader();w.writerows(completed)
                save(out/'status.json',state)
        assert len(completed)==10
        state.update(state='done',updated=stamp())
        save(out/'status.json',state)
    except BaseException as exc:
        state.update(state='failed',error=str(exc),updated=stamp())
        save(out/'status.json',state)
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',action='store_true')
    p.add_argument('--model',type=int)
    p.add_argument('--test',type=int)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    (worker if a.worker else queue)(a)


if __name__=='__main__':
    main()
