"""Standalone staged installer: never stop an active training/GPU task."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time


def journal(root,state,**data):
    target=root/'later_handoff.json';temp=target.with_suffix('.tmp')
    temp.write_text(json.dumps(dict(state=state,time=time.time(),**data),indent=2));temp.replace(target)


def controller_pids(root,machine):
    result=[]
    for file in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args=[v.decode() for v in file.read_bytes().split(b'\0') if v]
            if '-m' not in args or args[args.index('-m')+1]!='tools.style_matrix.generation2.runner':continue
            if args[args.index('--root')+1]!=str(root) or args[args.index('--machine')+1]!=machine:continue
            result.append(int(file.parent.name))
        except (OSError,ValueError,IndexError):pass
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--machine',choices=['4090','48'],required=True)
    p.add_argument('--root',required=True);p.add_argument('--archive',required=True);p.add_argument('--sha256',required=True)
    p.add_argument('--resident-pid',type=int,default=0);a=p.parse_args();root=Path(a.root)
    with open(f'/tmp/dit_later_watch_{a.machine}.lock','a+') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'later_plan.json').exists():return
        archive=Path(a.archive)
        if hashlib.sha256(archive.read_bytes()).hexdigest()!=a.sha256:raise ValueError('Staged archive changed')
        arm,step=('S9Pair',60000) if a.machine=='4090' else ('S11',50000)
        marker=root/'mechanisms'/f'{arm}_{step}'/'generator_swap/complete.json'
        deadline=json.loads((root/'clock.json').read_text())['started_unix']+64*3600
        journal(root,'waiting_for_completed_boundary',arm=arm,step=step)
        while not marker.exists():
            if time.time()>=deadline:journal(root,'expired');return
            time.sleep(15)
        lock=open('/tmp/dit_gpu0.lock','a+')
        while True:
            if time.time()>=deadline:journal(root,'expired');return
            # If the old single-horizon maturity has already started, preserve it.
            for item in (root/'runs'/arm/'checkpoints').glob('*.pt.done'):
                if int(item.name.split('.')[0])>step:
                    journal(root,'deferred_existing_maturity',checkpoint=str(item));return
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:time.sleep(1)
        lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory',
            '--format=csv,noheader,nounits'],text=True).strip()
        for line in lines.splitlines():
            pid,mb=map(int,line.split(','))
            if pid!=a.resident_pid or mb>1536:
                journal(root,'deferred_gpu_owner',owners=lines);return
        pids=controller_pids(root,a.machine)
        if len(pids)!=1:journal(root,'deferred_controller_identity',pids=pids);return
        # Global GPU lock proves this is a boundary, not a running scientific update.
        journal(root,'handoff_at_gpu_idle_boundary',old_controller=pids[0])
        os.kill(pids[0],signal.SIGTERM)
        until=time.time()+90
        while controller_pids(root,a.machine):
            if time.time()>=until:raise TimeoutError('Old controller did not exit')
            time.sleep(1)
        with tarfile.open(archive) as tar:
            members=tar.getmembers();workspace=Path.cwd().resolve()
            for member in members:
                target=(workspace/member.name).resolve()
                if not member.isfile() or not target.is_relative_to(workspace/'tools/style_matrix/later'):
                    raise ValueError('Unexpected staged source: '+member.name)
            tar.extractall(workspace,members=members)
        journal(root,'starting_later_controller',archive_sha256=a.sha256,first_stage_preserved=True)
        lock.close()
        os.execv(sys.executable,[sys.executable,'-u','-m','tools.style_matrix.later.runner',
            '--machine',a.machine,'--root',str(root),'--resident-pid',str(a.resident_pid)])


if __name__=='__main__':main()
