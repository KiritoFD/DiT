"""One 100-step microbatch search per injection family; no scientific training."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
from ..runtime.common import save


def main():
    p=argparse.ArgumentParser();p.add_argument('--machine',choices=['4090','48'],required=True)
    p.add_argument('--root',default='assets/style_matrix_20261026');p.add_argument('--resident-pid',type=int,default=0)
    a=p.parse_args();root=Path(a.root);out=root/'micro_search_b512'/a.machine;out.mkdir(parents=True,exist_ok=True)
    lock=open('/tmp/dit_gpu0.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    template=json.loads(next((root/'specs'/a.machine).glob('*.json')).read_text())
    choices=(192,224,256) if a.machine=='4090' else (384,448,512)
    selected=[]
    for injection in range(4):
        spec=dict(template);spec['config']=dict(template['config'])
        spec.update(arm=f'R2I{injection}_microsearch',machine=a.machine)
        spec['config'].update(style_matrix_repr=2,style_matrix_injection=injection,batch=512)
        spec_path=out/f'I{injection}.json';save(spec_path,spec);candidates=[]
        for micro in choices:
            result=out/f'I{injection}_b{micro}.json';log_path=result.with_suffix('.log')
            if result.exists():report=json.loads(result.read_text())
            else:
                owners=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory',
                    '--format=csv,noheader,nounits'],text=True).strip()
                for line in owners.splitlines():
                    pid,mb=map(int,line.split(','))
                    if pid!=a.resident_pid or mb>1536:raise RuntimeError('Unexpected GPU owner: '+owners)
                save(out/'status.json',dict(state='testing',injection=injection,microbatch=micro,optimizer_steps=100))
                with open(log_path,'w') as log:
                    rc=subprocess.call([sys.executable,'-u','-m','tools.style_matrix.probes.profile',
                        '--spec',str(spec_path),'--out',str(result),'--micro',str(micro),
                        '--steps','100','--capacity-only'],stdout=log,stderr=subprocess.STDOUT)
                if rc:
                    if 'out of memory' in log_path.read_text(errors='replace')[-7000:].lower():
                        save(out/f'I{injection}_b{micro}_oom.json',dict(kind='OOM',microbatch=micro));continue
                    save(out/'status.json',dict(state='failed',injection=injection,log=str(log_path)))
                    raise RuntimeError(str(log_path))
                report=json.loads(result.read_text())
            cap=21. if a.machine=='4090' else 42.
            if report['peak_reserved_GiB']<=cap:candidates.append(report)
        if not candidates:raise RuntimeError(f'I{injection}: no measured candidate below cap')
        best=max(candidates,key=lambda v:v['step_s'])
        selected.append(dict(injection=injection,chosen=best,
                             criterion='fastest measured below safe cap; no memory-fill target or gradient checkpointing',candidates=candidates))
        save(out/'selected.json',selected)
        print('FASTEST_MICROBATCH',injection,best['microbatch'],best['step_s'],flush=True)
    save(out/'status.json',dict(state='complete',families=4,scientific_training_started=False))


if __name__=='__main__':main()
