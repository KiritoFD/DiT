"""Finite GPU batch tests only. Does not launch any scientific training."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
from ..runtime.common import save


def main():
    p=argparse.ArgumentParser();p.add_argument('--machine',choices=['4090','48'],required=True)
    p.add_argument('--root',default='assets/style_matrix_20261026')
    p.add_argument('--resident-pid',type=int,default=0)
    p.add_argument('--selected-defaults',action='store_true')
    a=p.parse_args();root=Path(a.root);out=root/('batch_tests_b512' if a.selected_defaults else 'batch_tests')/a.machine;out.mkdir(parents=True,exist_ok=True)
    defaults={}
    if a.selected_defaults:
        defaults={v['injection']:v['chosen']['microbatch'] for v in json.loads((root/'micro_search_b512'/a.machine/'selected.json').read_text())}
    lock=open('/tmp/dit_gpu0.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock.seek(0);lock.truncate();lock.write(str(__import__('os').getpid()));lock.flush()
    summaries=[]
    for spec_path in sorted((root/'specs'/a.machine).glob('*.json')):
        spec=json.loads(spec_path.read_text());arm=spec['arm'];selected=None
        choices=(256,128,64)
        if a.selected_defaults:
            spec['config']['batch']=512
            default=defaults[spec['config']['style_matrix_injection']]
            choices=list(dict.fromkeys([default,192,128,64]))
            spec_path=out/f'{arm}_spec.json';save(spec_path,spec)
        for micro in choices:
            owners=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory',
                '--format=csv,noheader,nounits'],text=True).strip()
            for line in owners.splitlines():
                pid,mb=map(int,line.split(','))
                if pid!=a.resident_pid or mb>1536:raise RuntimeError('Unowned CUDA process: '+owners)
            result=out/f'{arm}_b{micro}.json';log_path=out/f'{arm}_b{micro}.log'
            if result.exists():report=json.loads(result.read_text())
            else:
                save(out/'status.json',dict(state='testing',arm=arm,microbatch=micro))
                command=[sys.executable,'-u','-m','tools.style_matrix.probes.profile','--spec',str(spec_path),
                         '--out',str(result),'--micro',str(micro),'--steps','20','--capacity-only']
                with open(log_path,'w') as log:
                    rc=subprocess.call(command,stdout=log,stderr=subprocess.STDOUT)
                if rc:
                    tail=log_path.read_text(errors='replace')[-6000:]
                    if 'out of memory' in tail.lower():
                        save(out/f'{arm}_b{micro}_failure.json',dict(kind='OOM',returncode=rc,log=str(log_path)))
                        continue
                    save(out/'status.json',dict(state='failed',arm=arm,log=str(log_path),returncode=rc))
                    raise RuntimeError(f'{arm}: implementation failure, see {log_path}')
                report=json.loads(result.read_text())
            cap=21. if a.machine=='4090' else 42.
            if report['peak_reserved_GiB']>cap and micro>64:continue
            selected=report;break
        if selected is None:raise RuntimeError(f'{arm}: no tested microbatch fits')
        summaries.append(selected);save(out/'selected.json',summaries)
        print('BATCH_SELECTED',arm,selected['microbatch'],selected['step_s'],selected['peak_reserved_GiB'],flush=True)
    assert len(summaries)==12,'Expected ten main cells and two cross-machine repeats'
    save(out/'status.json',dict(state='complete',cases=len(summaries),scientific_training_started=False))


if __name__=='__main__':main()
