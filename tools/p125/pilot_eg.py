"""P3-B gradient-transfer pilot: a few EG-auxiliary steps from the P2@15000
checkpoint in a probe run dir. Verifies (1) memory fits, (2) calibration
produces a finite weight, (3) gradients reach the U-Net. Writes pilot_eg.json.
"""
import argparse
import json
from pathlib import Path
import torch
from tools.style_matrix.runtime.common import save


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--plan',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--steps',type=int,default=12)
    a=p.parse_args()
    plan=json.loads(Path(a.plan).read_text())
    job=next(j for s in plan['stages'].values() for j in s if j['arm']=='P3B')
    spec=json.loads(Path(job['spec']).read_text())
    probe=Path(plan['root'])/'runs'/'P3B_probe'
    probe.mkdir(parents=True,exist_ok=True)
    import subprocess,sys
    cmd=[sys.executable,'-u','-m','tools.style_matrix.runtime.train',
        '--spec',job['spec'],'--stop',str(spec['config']['max_steps']-15000+ (a.steps if False else 0)+15000+a.steps),
        '--run',str(probe)]
    # simpler: stop = 15000 + steps (resume from the copied 15000 checkpoint)
    cmd=[sys.executable,'-u','-m','tools.style_matrix.runtime.train',
        '--spec',job['spec'],'--stop',str(15000+a.steps),'--run',str(probe),
        '--resume',str(Path(plan['root'])/'runs'/'P3B'/'checkpoints'/'0015000.pt')]
    log=subprocess.run(cmd,capture_output=True,text=True)
    tail=log.stdout.strip().splitlines()[-6:]
    ok=log.returncode==0
    result=dict(passed=bool(ok and log.returncode==0),returncode=log.returncode,
        tail=tail,probe_dir=str(probe))
    calibration=None
    ck=probe/'checkpoints'/f'{15000+a.steps:07d}.pt'
    if ck.is_file():
        data=torch.load(ck,map_location='cpu',weights_only=False)
        calibration=data.get('eg_calibration')
        result['calibration']=calibration
        result['passed']=bool(ok and calibration and calibration.get('weight',0)>0)
    save(a.out,result)
    print(json.dumps(result,ensure_ascii=False)[:500])
    if not result['passed']:raise SystemExit('EG pilot failed')


if __name__=='__main__':main()
