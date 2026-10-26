"""Fail-fast, resumable serial stages. Dry validation is the default."""
import argparse
import fcntl
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
from .common import save,sha


def completed_checkpoint(path):
    manifest=json.loads(Path(str(path)+'.done').read_text())
    if path.stat().st_size!=manifest['size'] or sha(path)!=manifest['sha256']:
        raise ValueError('Incomplete/corrupt checkpoint: '+str(path))
    return int(manifest['step'])


def latest(run):
    candidates=sorted((Path(run)/'checkpoints').glob('*.pt.done'),reverse=True)
    if not candidates:return None,0
    path=Path(str(candidates[0])[:-5]);return path,completed_checkpoint(path)


def verify_assets(plan):
    ready=json.loads(Path(plan['ready']).read_text())
    if ready['contract']!=plan['contract']:raise ValueError('Data/source contract changed')
    for item in ready['files']:
        if sha(item['path'])!=item['sha256']:raise ValueError('Changed input: '+item['path'])


def gpu_idle(resident):
    lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory',
        '--format=csv,noheader,nounits'],text=True).strip()
    for line in lines.splitlines():
        pid,mb=map(int,line.split(','))
        if pid!=resident or mb>1536:raise RuntimeError('Unexpected GPU owner: '+lines)


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True)
    p.add_argument('--stage',type=int,choices=[1,2,3],default=1);p.add_argument('--selection',default='')
    p.add_argument('--execute',action='store_true');p.add_argument('--resident-pid',type=int,default=0)
    a=p.parse_args();plan=json.loads(Path(a.plan).read_text());root=Path(plan['root'])
    jobs=plan['stages'][str(a.stage)]
    if a.stage>1:
        if not a.selection:raise ValueError('Promotions require an explicit reviewed selection file')
        selection=json.loads(Path(a.selection).read_text());chosen=selection[str(a.stage)][plan['machine']]
        limits=plan['promotion_limits'][str(a.stage)]
        if sum(v.startswith('S') for v in chosen)>limits['S'] or sum(v.startswith('J') for v in chosen)>limits['J']:
            raise ValueError('Selection exceeds the measured budget quota')
        known={v['arm'] for v in jobs}
        if len(chosen)!=len(set(chosen)) or not set(chosen)<=known:raise ValueError('Invalid promotion IDs')
        jobs=[v for v in jobs if v['arm'] in chosen]
        if not jobs:raise ValueError('Empty stage selection')
        joint={v for v in chosen if v.startswith('J')}
        if a.stage==2 and plan['machine']=='48' and joint!={'J0','J1','J2','J3'}:
            raise ValueError('All four J arms must reach the second boundary')
        if a.stage==3 and joint and (len(joint)!=2 or not joint&{'J0','J1'}):
            raise ValueError('Final J extension needs a matching frozen-decoder control')
    for job in jobs:
        if sha(job['spec'])!=job['spec_sha256']:raise ValueError('Spec changed: '+job['arm'])
    if not a.execute:
        print(json.dumps(dict(mode='dry',machine=plan['machine'],stage=a.stage,jobs=jobs,
            estimated_upper_hours=plan['estimated_upper_hours'],requires_ready=plan['ready']),indent=2));return
    verify_assets(plan)
    lock=open('/tmp/dit_gpu0.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    root.mkdir(parents=True,exist_ok=True);clock=root/'clock.json'
    if clock.exists():
        record=json.loads(clock.read_text())
        if record['plan_sha256']!=sha(a.plan):raise ValueError('Immutable execution plan changed')
        started=record['started_unix']
    else:started=time.time();save(clock,dict(started_unix=started,plan_sha256=sha(a.plan)))
    deadline=started+plan['wall_hours']*3600
    child=None;stopped=False
    def terminate(*_):
        nonlocal stopped
        stopped=True
        if child is not None and child.poll() is None:child.send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
    def call(command,log):
        nonlocal child
        gpu_idle(a.resident_pid)
        with log.open('a') as stream:
            child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
            while child.poll() is None:
                if time.time()>=deadline:terminate()
                time.sleep(2)
        rc=child.returncode;child=None
        if rc:
            save(root/'status.json',dict(state='stopped' if stopped or rc==75 else 'failed',
                returncode=rc,command=command,automatic_retry=False))
            raise SystemExit(rc)
        if stopped:raise SystemExit(75)
    for job in jobs:
        if stopped or time.time()>=deadline:raise SystemExit(75)
        run=root/'runs'/job['arm'];run.mkdir(parents=True,exist_ok=True)
        checkpoint,step=latest(run)
        if a.stage>1 and step<job['previous_stop']:raise ValueError('Previous stage incomplete: '+job['arm'])
        if step<job['stop']:
            save(root/'status.json',dict(state='training',stage=a.stage,arm=job['arm'],stop=job['stop']))
            command=[sys.executable,'-u','-m','tools.style_matrix.runtime.train',
                '--spec',job['spec'],'--stop',str(job['stop']),'--run',str(run)]
            if checkpoint:command+=['--resume',str(checkpoint)]
            call(command,run/'console.log');checkpoint,step=latest(run)
            if step!=job['stop']:raise ValueError('Stage did not reach its boundary')
        stage_checkpoint=run/'checkpoints'/f"{job['stop']:07d}.pt"
        completed_checkpoint(stage_checkpoint)
        for protocol in plan['evaluations']:
            out=run/'evaluations'/f"{job['stop']:07d}"/protocol['name']
            expected=dict(checkpoint_sha256=sha(stage_checkpoint),csv_sha256=sha(protocol['csv']))
            if (out/'complete.json').exists():
                done=json.loads((out/'complete.json').read_text())
                if any(done.get(k)!=v for k,v in expected.items()):raise ValueError('Eval identity changed')
                if not (out/'result.json').exists():raise ValueError('Completed eval missing metrics')
                continue
            out.mkdir(parents=True,exist_ok=True)
            save(root/'status.json',dict(state='evaluating',stage=a.stage,arm=job['arm'],protocol=protocol['name']))
            call([sys.executable,'-u','-m','tools.style_matrix.runtime.evaluate',
                '--checkpoint',str(stage_checkpoint),'--csv',protocol['csv'],'--out',str(out),
                '--writer-space',protocol['writer_space']],out/'console.log')
        save(run/f'stage{a.stage}.json',dict(state='complete',step=job['stop'],checkpoint=str(stage_checkpoint),
            checkpoint_sha256=sha(stage_checkpoint)))
    save(root/'status.json',dict(state='stage_complete',stage=a.stage,automatic_promotion=False))


if __name__=='__main__':main()
