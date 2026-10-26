"""Finite J probes: compare two image microbatches, then verify the other arms."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
from ..runtime.common import save


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--fm-micro',type=int,default=448)
    p.add_argument('--cap',type=float,default=42.);p.add_argument('--steps',type=int,default=5)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    lock=open('/tmp/dit_gpu0.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    candidates=[];reports={}
    def run(arm,micro):
        target=out/f'{arm}_image{micro}.json';log=target.with_suffix('.log')
        if target.exists():return json.loads(target.read_text())
        owners=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory',
            '--format=csv,noheader,nounits'],text=True).strip()
        if owners:raise RuntimeError('Foreign CUDA owner: '+owners)
        save(out/'status.json',dict(state='testing',arm=arm,image_micro=micro,capacity_only=True))
        with log.open('w') as stream:
            rc=subprocess.call([sys.executable,'-u','-m','tools.style_matrix.joint.profile',
                '--arm',arm,'--fm-micro',str(a.fm_micro),'--image-micro',str(micro),
                '--steps',str(a.steps),'--out',str(target)],stdout=stream,stderr=subprocess.STDOUT)
        if rc:
            if 'out of memory' in log.read_text(errors='replace')[-7000:].lower():
                save(target.with_name(target.stem+'_failure.json'),dict(kind='OOM',returncode=rc));return None
            save(out/'status.json',dict(state='failed',arm=arm,log=str(log),returncode=rc))
            raise RuntimeError(str(log))
        return json.loads(target.read_text())
    for micro in (8,4):
        r=run('J3',micro)
        if r and r['peak_reserved_GiB']<=a.cap:candidates.append(r)
    if not candidates:raise RuntimeError('Neither J3 candidate fits the safety cap')
    best=max(candidates,key=lambda r:r['step_s']);reports['J3']=best
    for arm in ('J0','J1','J2'):
        r=run(arm,best['image_microbatch'])
        if not r or r['peak_reserved_GiB']>a.cap:raise RuntimeError(f'{arm}: memory preflight failed')
        reports[arm]=r
    save(out/'selected.json',dict(arms=reports,criterion='fastest measured safe J3 image microbatch; shared across arms',
        candidates=candidates,scientific_training_started=False))
    save(out/'status.json',dict(state='complete',arms=4,capacity_only=True,scientific_training_started=False))


if __name__=='__main__':main()
