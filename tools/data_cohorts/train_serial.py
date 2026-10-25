"""Frozen Top20/10/5/2 U114 queue; fail-stop, independent of the SSH session."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo


def stamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def prepare(args):
    root, launch = Path(args.root), Path(args.launch)
    ready = json.loads((root/'ready_final/READY.json').read_text())
    data = root/'datasets_final'
    assert digest(data/'manifest.json') == ready['dataset_manifest_sha256']
    manifest = json.loads((data/'manifest.json').read_text())
    for size, level in manifest['levels'].items():
        for name, expected in level['files'].items():
            assert digest(data/f'top{size}'/name) == expected, name
    launch.mkdir(parents=True, exist_ok=False)
    jobs = []
    from src.unet.io import Config
    for size in (20, 10, 5, 2):
        template = root/f'ready_final/top{size}_u114.template.json'
        assert digest(template) == ready['configs'][template.name]
        config = json.loads(template.read_text())
        config.update(max_steps=args.steps, lr_decay_steps=args.steps,
                      stop_at_step=args.stop_steps, lr=args.lr,
                      output=f'assets/results/clean_cohorts_20261021/{launch.name}/top{size}_u114')
        assert not Path(config['output']).exists(), config['output']
        Config(**config).validate()
        path = launch/f'top{size}.json'
        save(path, config)
        inputs = {str(p): digest(p) for p in (data/f'top{size}').rglob('*') if p.is_file()}
        jobs.append(dict(cohort=size, config=str(path), config_sha256=digest(path),
                         inputs=inputs, output=config['output'], log=str(launch/f'top{size}.log')))
    source = [Path('src/unet')/n for n in ('model.py','blocks.py','io.py','train.py')]
    save(launch/'queue.json', dict(created=stamp(), python=args.python, max_steps=args.steps,
        stop_at_step=args.stop_steps, lr=args.lr,
        order=[20,10,5,2], data_root=str(root), ready_sha256=digest(root/'ready_final/READY.json'),
        source_sha256={str(p):digest(p) for p in source}, jobs=jobs))
    save(launch/'status.json', dict(state='prepared', updated=stamp(), completed=[]))
    print(launch, flush=True)


def run(args):
    launch = Path(args.launch)
    # Exclusive lock intentionally remains after failure; never silently replay a queue.
    with (launch/'started.lock').open('x') as f:
        f.write(f'{os.getpid()} {stamp()}\n')
    queue = json.loads((launch/'queue.json').read_text())
    for name, expected in queue['source_sha256'].items():
        assert digest(name) == expected, f'Source changed: {name}'
    status = dict(state='starting', runner_pid=os.getpid(), started=stamp(), completed=[])
    try:
        for job in queue['jobs']:
            assert digest(job['config']) == job['config_sha256']
            for name, expected in job['inputs'].items():
                assert digest(name) == expected, f'Frozen input changed: {name}'
            status.update(state='running', cohort=job['cohort'], log=job['log'], updated=stamp())
            with open(job['log'], 'xb') as log:
                child = subprocess.Popen([queue['python'], '-u', '-m', 'src.unet.train',
                                          '--config', job['config']], stdout=log, stderr=subprocess.STDOUT)
                status['child_pid'] = child.pid
                save(launch/'status.json', status)
                result = child.wait()
            if result:
                raise RuntimeError(f"Top{job['cohort']} exited {result}; see {job['log']}")
            # Success must include the final budget, not merely a clean process exit.
            with open(job['log'], 'rb') as f:
                f.seek(max(0, os.fstat(f.fileno()).st_size-32768))
                tail = f.read().decode('utf-8', errors='replace')
            target = queue.get('stop_at_step') or queue['max_steps']
            if f"DONE checkpoint_step={target}" not in tail:
                raise RuntimeError('Training ended without the requested final checkpoint')
            status['completed'].append(job['cohort'])
            save(launch/'status.json', status)
        status.update(state='done', updated=stamp())
        save(launch/'status.json', status)
    except BaseException as exc:
        status.update(state='failed', error=str(exc), updated=stamp())
        save(launch/'status.json', status)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare','run'))
    parser.add_argument('--launch', required=True)
    parser.add_argument('--root', default='assets/data_cohorts/top20_20261021_v1')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--steps', type=int, default=70000)
    parser.add_argument('--stop-steps', type=int, default=0)
    parser.add_argument('--lr', type=float, default=6e-5)
    args = parser.parse_args()
    (prepare if args.action == 'prepare' else run)(args)


if __name__ == '__main__':
    main()
