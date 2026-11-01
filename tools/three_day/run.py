"""One owned process per machine, serial subprocesses, fail closed, explicit artifacts."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .contracts import save,verify_inputs


class Queue:
    def __init__(self,root,retry_failed=False):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.lock=(self.root/'queue.lock').open('w');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        p=self.root/'status.json'
        self.status=json.loads(p.read_text()) if p.exists() else {'phases':{},'started':time.strftime('%F %T')}
        self.status.update(pid=os.getpid(),state='running');self.flush()
        self.retry_failed=retry_failed

    def flush(self):save(self.root/'status.json',self.status)

    def command(self,name,cmd,required=()):
        old=self.status['phases'].get(name,{})
        if old.get('state')=='failed' and self.retry_failed:
            try:os.kill(old['child_pid'],0)
            except ProcessLookupError:pass
            else:raise RuntimeError('Refuse retry while the old child PID is live')
            self.status.setdefault('failed_attempts',[]).append(dict(phase=name,**old))
            previous=self.root/'logs'/(name+'.log')
            if previous.exists():previous.rename(previous.with_suffix('.failed_'+str(int(time.time()))+'.log'))
            old={}
        if old.get('state')=='done':
            if not all(Path(p).exists() for p in required):raise FileNotFoundError('Completed artifact missing: '+name)
            return
        if old.get('state') in ('running','failed'):
            raise RuntimeError('Unfinished phase requires explicit full-resume audit, not an automatic restart: '+name)
        log=self.root/'logs'/(name+'.log');log.parent.mkdir(exist_ok=True)
        entry=dict(state='running',command=cmd,started=time.strftime('%F %T'),started_unix=time.time())
        self.status['phases'][name]=entry;self.status['current']=name;self.flush()
        with log.open('w') as stream:
            child=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT)
            entry['child_pid']=child.pid;self.flush();code=child.wait()
        if code or not all(Path(p).exists() for p in required):
            entry.update(state='failed',returncode=code);self.status['state']='failed';self.flush()
            raise RuntimeError(f'{name} failed, later phases not launched; see {log}')
        entry.update(state='done',finished=time.strftime('%F %T'),seconds=time.time()-entry['started_unix'])
        self.flush()


def final_checkpoint(root,arm,stage,step):
    files=list((root/arm/stage/'runs').glob(f'*/checkpoints/{step:07d}.pt'))
    if len(files)!=1:raise ValueError(f'Expected one final checkpoint: {arm}/{stage}/{step}, found {files}')
    path=files[0]
    if not path.with_suffix('.pt.done').exists():raise FileNotFoundError('Checkpoint completion marker missing: '+str(path))
    transfer=json.loads((path.parents[1]/'stage_transfer.json').read_text())
    if not transfer['all_tensors_equal']:raise ValueError('Stage transfer was not strict')
    return str(path)


def day1(queue):
    from .prepare import prepare
    manifest=prepare(queue.root)
    for arm in ('A0','A3','A1','A2'):
        for stage,step in [('p1',10000),('p2',20000)]:
            verify_inputs(manifest)
            path=queue.root/arm/(stage+'.json');config=json.loads(path.read_text())
            if stage=='p2':
                config['initialize_from']=final_checkpoint(queue.root,arm,'p1',10000);save(path,config)
            queue.command(arm+'_'+stage,[sys.executable,'-u','-m','src.unet.train','--config',str(path)])
            final_checkpoint(queue.root,arm,stage,step)
    queue.status.update(state='day1_complete',finished_day1=time.strftime('%F %T'));queue.flush()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='assets/three_day_20261026')
    p.add_argument('--host',choices=['4090','48'],required=True)
    p.add_argument('--retry-failed',action='store_true',help='Explicit retry only after a verified terminal failure and repair')
    p.add_argument('--detach',action='store_true')
    a=p.parse_args()
    if a.detach:
        root=Path(a.root);root.mkdir(parents=True,exist_ok=True)
        with (root/'detached.log').open('a') as stream:
            child=subprocess.Popen([sys.executable,'-u','-m','tools.three_day.run',
                *[v for v in sys.argv[1:] if v!='--detach']],stdin=subprocess.DEVNULL,stdout=stream,
                stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
        print('DETACHED_QUEUE_PID',child.pid,flush=True);return
    queue=Queue(a.root,a.retry_failed)
    if a.host=='4090':day1(queue)
    else:
        from .audits.run import run
        run(queue)


if __name__=='__main__':main()
