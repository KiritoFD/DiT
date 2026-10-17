"""Checkpoint-safe handover and fail-fast 60k connectivity queue (Linux)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]


def write_status(folder, phase, **fields):
    record = dict(phase=phase, time=time.strftime('%Y-%m-%d %H:%M:%S'), **fields)
    target = folder / 'status.json'
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2), encoding='utf-8')
    temporary.replace(target)
    print(json.dumps(record), flush=True)


def handover(args, folder):
    source = Path(args.checkpoint).resolve()
    process = Path('/proc') / str(args.stop_pid)
    expected = Path(args.stop_cwd).resolve()
    deadline = time.monotonic() + args.wait_timeout
    write_status(folder, 'waiting_checkpoint', checkpoint=str(source), old_pid=args.stop_pid)
    while True:
        if not process.exists():
            raise RuntimeError('Original training exited before a safe handover')
        command = (process / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
        if (process / 'cwd').resolve() != expected or '-m src.unet.train' not in command:
            raise RuntimeError('PID identity changed; refusing to stop it')
        # Waiting for the NEXT logged update also lets checkpoint-time evaluation finish.
        tail = Path(args.stop_log).read_text(encoding='utf-8')[-65536:]
        steps = [int(s) for s in re.findall(r'\(step=(\d+)\)', tail)]
        if source.is_file() and source.with_suffix('.pt.done').is_file() and steps and max(steps) > args.handover_step:
            break
        if time.monotonic() > deadline:
            raise TimeoutError('Checkpoint handover deadline exceeded; old training untouched')
        time.sleep(15)
    if os.getpgid(args.stop_pid) != args.stop_pid:
        raise RuntimeError('Training does not own its process group')
    os.killpg(args.stop_pid, signal.SIGTERM)
    for _ in range(60):
        busy = subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
        if not busy:
            return source
        time.sleep(1)
    raise RuntimeError('GPU did not become idle after SIGTERM; queue aborted')


def follow_existing(args,folder):
    """Wait for an already-running baseline, without signalling or restarting it."""
    process=Path('/proc')/str(args.after_pid)
    final=Path(args.after_run).resolve()/'checkpoints'/f'{args.max_steps:07d}.pt'
    write_status(folder,'waiting_baseline',pid=args.after_pid,final_checkpoint=str(final))
    while process.exists() and (process/'stat').read_text().split()[2]!='Z':
        command=(process/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        if (process/'cwd').resolve()!=Path(args.after_cwd).resolve() or '-m src.unet.train' not in command:
            raise RuntimeError('Baseline PID identity changed')
        time.sleep(30)
    if not final.is_file() or not final.with_suffix('.pt.done').is_file():
        raise RuntimeError('Baseline exited without its final complete checkpoint')
    if 'DONE checkpoint_step=' not in (Path(args.after_run)/'train.log').read_text()[-8192:]:
        raise RuntimeError('Baseline did not report clean training completion')
    with final.open('rb') as handle:
        digest=hashlib.file_digest(handle,'sha256').hexdigest()
    marker=json.loads(final.with_suffix('.pt.done').read_text())
    if marker['sha256']!=digest or marker['step']!=args.max_steps:
        raise RuntimeError('Baseline final checkpoint identity failed')
    return Path(args.checkpoint).resolve()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--stop-pid', type=int)
    p.add_argument('--stop-cwd')
    p.add_argument('--stop-log')
    p.add_argument('--after-pid',type=int)
    p.add_argument('--after-cwd')
    p.add_argument('--after-run')
    p.add_argument('--jobs',nargs='+',choices=['baseline_resume','A_control','E_combined','F_combined_writer'],
                   default=['baseline_resume','A_control','E_combined'])
    p.add_argument('--handover-step', type=int, required=True)
    p.add_argument('--max-steps', type=int, required=True)
    p.add_argument('--wait-timeout', type=int, default=7200)
    p.add_argument('--log-dir', required=True)
    p.add_argument('--run', action='store_true')
    args = p.parse_args()
    if bool(args.stop_pid)==bool(args.after_pid):p.error('Specify exactly one of --stop-pid or --after-pid')
    if args.stop_pid and not (args.stop_cwd and args.stop_log):p.error('Stopping requires cwd and log')
    if args.after_pid and not (args.after_cwd and args.after_run):p.error('Following requires cwd and run')
    if args.max_steps <= args.handover_step:
        raise ValueError('max-steps must exceed the handover checkpoint')
    base = json.loads((ROOT / 'src/unet/configs/connectivity/A_control.json').read_text())
    base.update(max_steps=args.max_steps, lr_decay_steps=args.max_steps, checkpoint_steps=[], stop_at_step=0)
    # Control matches the expanded model's unavoidable optimizer/EMA restart.
    combined=dict(skip_gate_scales=[0],decoder_res_blocks=[3,3,2],std_gate_scales=[0,1,2])
    jobs = [('baseline_resume', dict(requires_warm_start=False, ema_start_step=20000), '--resume'),
            ('A_control', {}, '--warm-start'),
            ('E_combined', combined, '--warm-start'),
            ('F_combined_writer',dict(combined,writer_film_rank=16),'--warm-start')]
    plan = []
    folder = Path(args.log_dir).resolve()
    for name, changes, mode in jobs:
        if name not in args.jobs:continue
        config = dict(base, **changes)
        config['output'] = f'assets/results/unet_connectivity_{args.max_steps}/{name}'
        plan.append(dict(name=name, config=config, mode=mode))
    print(json.dumps(plan, indent=2), flush=True)
    if not args.run:
        return
    folder.mkdir(parents=True, exist_ok=False)
    (folder/'queued_plan.json').write_text(json.dumps(dict(source=args.checkpoint,jobs=plan),indent=2))
    os.chdir(ROOT)
    try:
        source = follow_existing(args,folder) if args.after_pid else handover(args, folder)
        with source.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        marker = json.loads(source.with_suffix('.pt.done').read_text())
        if marker['sha256'] != digest or marker['step'] != args.handover_step:
            raise RuntimeError('Handover checkpoint identity failed')
        (folder / 'plan.json').write_text(json.dumps(dict(source=str(source), sha256=digest, jobs=plan), indent=2))
        for job in plan:
            busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
            if busy:
                raise RuntimeError('GPU occupied before queue job: ' + busy)
            config_path = folder / (job['name'] + '.json')
            config_path.write_text(json.dumps(job['config'], indent=2))
            command = [sys.executable, '-u', '-m', 'src.unet.train', '--config', str(config_path), job['mode'], str(source)]
            write_status(folder, 'running', job=job['name'], command=command)
            with (folder / (job['name'] + '.log')).open('x') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            # Never advance after an early clean exit or missing final save.
            runs = sorted((ROOT / job['config']['output']).glob('*-Unet-flow'))
            final = runs[-1] / 'checkpoints' / f'{args.max_steps:07d}.pt'
            if not final.is_file() or not final.with_suffix('.pt.done').is_file():
                raise RuntimeError('Job did not produce its final complete checkpoint: ' + str(final))
        write_status(folder, 'complete')
    except BaseException as error:
        write_status(folder, 'failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
