"""Measure peak memory and steady-state throughput for one D arm.

Two things this deliberately does NOT do:

* search over batch candidates — the largest intended configuration already fit
  (D1 20.25 GiB, D2 19.72 GiB reserved), so a search would only pay `torch.compile`
  warm-up over and over in fresh processes;
* trust the in-loop `step_s` on a short run — it is cumulative from step 0 and
  therefore still contains the compile warm-up. The steady-state rate is derived
  from the last two records instead.
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run_once(spec_path, work, stop, python, cache_dir):
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    cmd = [python, '-u', '-m', 'tools.style_matrix.runtime.train',
           '--spec', str(spec_path), '--stop', str(stop), '--run', str(work / 'run')]
    env = dict(**__import__('os').environ)
    if cache_dir:
        env['TORCHINDUCTOR_CACHE_DIR'] = cache_dir
    print('  $', ' '.join(cmd), flush=True)
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env)
    if p.returncode != 0:
        return None, (p.stdout or '')[-2500:] + (p.stderr or '')[-2500:]
    recs = []
    for line in (p.stdout or '').splitlines():
        line = line.strip()
        if line.startswith('{'):
            try:
                recs.append(json.loads(line))
            except Exception:
                pass
    return recs, (p.stdout or '')[-800:]


def steady_rate(recs):
    """Marginal seconds/step between the last two records (drops compile warm-up)."""
    if len(recs) < 2:
        return None, None
    a, b = recs[-2], recs[-1]
    ta, tb = a['step'] / a['step_s'], b['step'] / b['step_s']
    if b['step'] <= a['step'] or tb <= ta:
        return None, None
    return (tb - ta) / (b['step'] - a['step']), (a['step'], b['step'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--specs', default='assets/d_20261101/specs')
    ap.add_argument('--work', default='assets/d_20261101/preflight')
    ap.add_argument('--stop', type=int, default=30)
    ap.add_argument('--budget', type=float, default=21.0)
    ap.add_argument('--python', default='/opt/conda/envs/cu121/bin/python')
    ap.add_argument('--arms', default='D1,D2,D3')
    ap.add_argument('--cache', default='assets/d_20261101/inductor_cache')
    a = ap.parse_args()

    Path(a.cache).mkdir(parents=True, exist_ok=True)
    out, ok = {}, True
    for arm in [x.strip() for x in a.arms.split(',')]:
        spec_path = Path(a.specs) / f'{arm}.json'
        spec = json.loads(spec_path.read_text(encoding='utf-8'))
        print(f'=== {arm}', flush=True)
        recs, tail = run_once(spec_path, Path(a.work) / arm, a.stop, a.python, a.cache)
        if recs is None:
            print(f'  FAILED\n{tail[-1500:]}', flush=True)
            out[arm] = dict(failed=True, tail=tail[-1500:])
            ok = False
            continue
        last = recs[-1]
        rate, window = steady_rate(recs)
        res = dict(peak_GiB=round(last.get('peak_GiB', 0), 2),
                   peak_reserved_GiB=round(last.get('peak_reserved_GiB', 0), 2),
                   fm=last.get('fm'), decoder_grad_norm=last.get('decoder_grad_norm'),
                   auxiliary=last.get('auxiliary'),
                   step_s_steady=None if rate is None else round(rate, 4),
                   window=window,
                   fits=last.get('peak_reserved_GiB', 99) <= a.budget)
        if rate:
            res['steps_2h'] = int(7200 / rate)
            res['steps_3h'] = int(10800 / rate)
        out[arm] = res
        print('  ' + json.dumps(res), flush=True)
        if not res['fits']:
            ok = False

    dst = Path(a.work) / 'preflight.json'
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
    print('\n=== summary ===')
    for arm, v in out.items():
        if v.get('failed'):
            print(f'  {arm}: FAILED')
        else:
            print(f'  {arm}: reserved={v["peak_reserved_GiB"]:5.2f}GiB  '
                  f'{v["step_s_steady"]}s/step  fits={v["fits"]}  '
                  f'2h={v.get("steps_2h")} 3h={v.get("steps_3h")}')
    print('wrote', dst)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
