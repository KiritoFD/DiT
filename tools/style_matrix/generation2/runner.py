"""Resumable 24/48/64h controller: immutable per-job plans and explicit gates."""
import argparse
import copy
import csv
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from .plan import read,recipes,protocols,seal
from .metrics import measure,decoder_pair
from tools.style_matrix.runtime.common import save,sha
from tools.style_matrix.runtime.serial import gpu_idle,verify_assets


class Controller:
    def __init__(self,machine,root,resident=0):
        self.machine=machine;self.root=Path(root);self.resident=resident;self.child=None
        self.root.mkdir(parents=True,exist_ok=True)
        self.owner=open(f'/tmp/dit_gen2_{machine}.lock','a+');fcntl.flock(self.owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (self.root/'failure.json').exists():
            history=self.root/'incidents';history.mkdir(exist_ok=True)
            (self.root/'failure.json').replace(history/f'recovered_{int(time.time())}.json')
        if not (self.root/'clock.json').exists():save(self.root/'clock.json',dict(started_unix=time.time(),budget_hours=64))
        self.started=read(self.root/'clock.json')['started_unix'];self.deadline=self.started+64*3600
        signal.signal(signal.SIGTERM,self.stop);signal.signal(signal.SIGINT,self.stop)
        if not (self.root/'ready.json').exists():
            self.specs=recipes(self.root,machine)
            self.ps=protocols(self.root,self.specs['S9' if machine=='4090' else 'S11'],'S')
            self.pj=protocols(self.root,self.specs['J4'],'J') if machine=='48' else []
            save(self.root/'protocols.json',dict(S=self.ps,J=self.pj));seal(self.root,machine)
        else:
            self.specs={p.stem:read(p) for p in (self.root/'specs').glob('*.json')}
            proto=read(self.root/'protocols.json');self.ps=proto['S'];self.pj=proto['J']
        self.ready=self.root/'ready.json'
        first=Path('assets/style_matrix_20261028/generation2')/machine/'execution/runs'
        old=Path('assets/style_matrix_20261027/matrix64')/machine/'execution/runs'
        for arm in ('J4','J5','J6','J7') if machine=='48' else ('S0','S4'):
            source=(first if arm.startswith('J') else old)/arm
            target=self.root/'runs'/arm
            target.parent.mkdir(parents=True,exist_ok=True)
            if source.exists() and not target.exists():target.symlink_to(source.resolve(),target_is_directory=True)
        matches=list(Path('assets/style_matrix_20261027').rglob('writer_proto.pt'))
        if not matches:raise FileNotFoundError('Existing audited Writer prototype bank not found')
        self.writer_cache=matches[0]

    def stop(self,*_):
        if self.child is not None and self.child.poll() is None:self.child.send_signal(signal.SIGTERM)
        raise SystemExit(75)

    def phase(self,number,message):
        save(self.root/'status.json',dict(phase=number,state='running',detail=message,
             elapsed_hours=(time.time()-self.started)/3600,deadline_unix=self.deadline))
        if time.time()>=self.deadline:raise SystemExit(75)

    def call(self,command,log,gpu=False):
        if time.time()>=self.deadline:raise SystemExit(75)
        verify_assets(dict(ready=str(self.ready),contract=read(self.ready)['contract']))
        lock=None
        if gpu:
            lock=open('/tmp/dit_gpu0.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);gpu_idle(self.resident)
        try:
            with Path(log).open('a') as stream:
                self.child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
                while self.child.poll() is None:
                    if time.time()>=self.deadline:self.child.send_signal(signal.SIGTERM)
                    for hour in (24,48,64):
                        snapshot=self.root/f'budget_{hour}h.json'
                        if time.time()>=self.started+hour*3600 and not snapshot.exists():
                            save(snapshot,dict(hours=hour,status=read(self.root/'status.json'),command=command,
                                completed_scores=[str(p) for p in (self.root/'tasks').glob('*/score.json')]))
                    time.sleep(2)
                rc=self.child.returncode;self.child=None
            if rc==75 and time.time()>=self.deadline:
                save(self.root/'budget_complete.json',dict(hours=64,last_log=str(log)));raise SystemExit(75)
            if rc:raise RuntimeError(f'Command failed ({rc}); preserved log: {log}')
        finally:
            if lock is not None:lock.close()

    def job(self,key,stop=None,fork=None,final=False):
        spec=self.specs[key];arm=spec['arm'];stop=stop or spec['config']['max_steps']
        family='J' if arm.startswith('J') else 'S';proto=self.pj if family=='J' else self.ps
        chosen=proto if final else [proto[0],proto[-1]]
        # Preserve the original validation metric alongside the unified protocol.
        if not final:chosen=[dict(name='original_protocol',csv=spec['config']['eval_csv'],
                                  writer_space='raw')]+chosen
        tag=f'{key}_{stop}_{"final" if final else "screen"}'
        task=self.root/'tasks'/tag;task.mkdir(parents=True,exist_ok=True)
        runs=task/'runs'
        if not runs.exists():runs.symlink_to((self.root/'runs').resolve(),target_is_directory=True)
        job=dict(arm=arm,spec=str(self.root/'specs'/f'{key}.json'),spec_sha256=sha(self.root/'specs'/f'{key}.json'),
                 stop=stop,previous_stop=0)
        if fork:job['fork_from']=str(self.root/'runs'/fork/'checkpoints/0050000.pt')
        plan=dict(machine=self.machine,root=str(task),ready=str(self.ready),contract=read(self.ready)['contract'],
             wall_hours=64,deadline_unix=self.deadline,estimated_upper_hours=None,stages={'1':[job]},evaluations=chosen)
        path=task/'plan.json'
        if path.exists() and read(path)!=plan:raise ValueError('Immutable job plan changed: '+tag)
        if not path.exists():save(path,plan)
        self.call([sys.executable,'-u','-m','tools.style_matrix.runtime.serial','--plan',str(path),'--execute',
                   '--resident-pid',str(self.resident)],task/'serial.log')
        folder=self.root/'runs'/arm/'evaluations'/f'{stop:07d}'
        normal=measure(folder/proto[0]['name'],proto[0],self.writer_cache)
        panel=measure(folder/proto[-1]['name'],proto[-1],self.writer_cache)
        score=dict(normal,direction=panel['direction'],direction_protocol='shared_train_panel_diagnostic_only')
        save(task/'score.json',score)
        if final:
            for item in proto[1:-1]:measure(folder/item['name'],item,self.writer_cache)
        self.mechanisms(arm,stop,proto[-1],family)
        return score

    def mechanisms(self,arm,stop,protocol,family):
        checkpoint=self.root/'runs'/arm/'checkpoints'/f'{stop:07d}.pt'
        out=self.root/'mechanisms'/f'{arm}_{stop}';out.mkdir(parents=True,exist_ok=True)
        if family=='S':
            if self.specs[arm]['config'].get('writer_velocity') and not (out/'velocity.json').exists():
                self.call([sys.executable,'-u','-m','tools.style_matrix.generation2.probes',
                    '--checkpoint',str(checkpoint),'--csv',protocol['csv'],'--out',str(out/'velocity.json')],out/'velocity.log',gpu=True)
            variants=[('generator_swap',['--writer-swap','generator'])]
        else:
            variants=[('original_decoder',['--original-decoder'])]
            if arm in ('J5','J6','J7'):variants.append(('decoder_swap',['--writer-swap','decoder']))
        for name,extra in variants:
            target=out/name
            if not (target/'complete.json').exists():
                self.call([sys.executable,'-u','-m','tools.style_matrix.runtime.evaluate','--checkpoint',str(checkpoint),
                    '--out',str(target),'--csv',protocol['csv'],'--writer-space','raw']+extra,out/f'{name}.log',gpu=True)

    def run(self):
        if self.machine=='4090':self.run_s()
        else:self.run_j()
        self.phase(3,'All scheduled scientific jobs completed')
        save(self.root/'complete.json',dict(completed_unix=time.time(),elapsed_hours=(time.time()-self.started)/3600))
        save(self.root/'status.json',dict(phase=3,state='complete',elapsed_hours=(time.time()-self.started)/3600))

    def run_s(self):
        self.phase(1,'S9 cold 50k, then matched FM/Pair forks to 60k')
        cold=self.job('S9');fm=self.job('S9FM',fork='S9');pair=self.job('S9Pair',fork='S9')
        bad=(pair['content']['ssim']<fm['content']['ssim']-.01 or pair['content']['lpips']>fm['content']['lpips']+.01)
        direction=lambda x:x['direction']['cosine'] if x['direction']['cosine'] is not None else -1.
        reject=bad and pair['writer']['margin_mean']<=fm['writer']['margin_mean'] and direction(pair)<=direction(fm)
        save(self.root/'pair_gate.json',dict(reject_pair_extension=reject,FM=fm,Pair=pair))
        self.phase(2,'S10 cold 50k, then mature S9 control/Pair')
        s10=self.job('S10')
        baselines={arm:self.job(arm,stop=50000) for arm in ('S0','S4')}
        base0=baselines['S0']
        def degraded(value):
            return (value['content']['ssim']<base0['content']['ssim']-.02 and
                    value['content']['lpips']>base0['content']['lpips']+.015 and
                    value['writer']['margin_mean']<=base0['writer']['margin_mean'])
        if degraded(cold) and degraded(s10):
            best='S9' if cold['content']['lpips']<=s10['content']['lpips'] else 'S10'
            save(self.root/'capacity_gate.json',dict(both_degraded=True,independent_seed=best+'_seed91'))
            self.phase(3,'Both new cold models degraded: independent seed instead of old-baseline extension')
            self.job(best,final=True);self.job(best+'_seed91',final=True);return
        fm=self.job('S9FM_75')
        if not reject:pair=self.job('S9Pair_75')
        self.phase(3,'S10 75k, equal-exposure baseline, independent seeds and mechanisms')
        self.job('S10' if degraded(s10) else 'S10_75',final=True)
        chosen='S4' if fm['writer']['margin_mean']>=base0['writer']['margin_mean']+.01 else 'S0'
        save(self.root/'baseline_selection.json',dict(chosen=chosen,rule='Writer margin +.01 over S0 => S4; otherwise S0'))
        self.job(chosen+'_75',final=True);self.job('S9FM_75',final=True)
        if not reject:self.job('S9Pair_75',final=True)

    def run_j(self):
        self.phase(1,'J4/J5/J6/J7 independent Deep85 forks, each 5k')
        scores={arm:self.job(arm) for arm in ('J4','J5','J6','J7')}
        pair=decoder_pair(scores);save(self.root/'decoder_gate.json',dict(selected_pair=pair,scores=scores,
            rule='Content guard + independent Writer gain + positive grouped GT direction; otherwise S11'))
        self.phase(2,'Evidence-selected decoder pair to 12k; independent S11 cold 50k')
        for arm in pair:self.job(arm+'_12',final=True)
        self.job('S11')
        self.phase(3,'S11 mature 75k and an independent 50k seed')
        self.job('S11_75',final=True);self.job('S11_seed91',final=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--machine',choices=['4090','48'],required=True)
    p.add_argument('--root',required=True);p.add_argument('--resident-pid',type=int,default=0)
    p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args();controller=Controller(a.machine,a.root,a.resident_pid)
    if a.prepare_only:
        print(json.dumps(dict(specs=list(controller.specs),protocols=controller.ps+controller.pj)),flush=True);return
    try:controller.run()
    except SystemExit as exc:
        save(controller.root/'status.json',dict(state='budget_complete' if time.time()>=controller.deadline else 'stopped',
             returncode=exc.code,deadline_unix=controller.deadline));raise
    except BaseException as exc:
        save(controller.root/'status.json',dict(state='failed',error=repr(exc)))
        save(controller.root/'failure.json',dict(error=repr(exc),time=time.time(),automatic_retry=False));raise


if __name__=='__main__':main()
