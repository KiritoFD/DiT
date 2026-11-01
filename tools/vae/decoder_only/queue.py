"""One GPU owner: finish existing T eval, smoke T2, train T2 -> V0 -> V1."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
P2='assets/u100_p1p2_20261025/p2_runs/20261025-152857-Unet-flow/checkpoints/0020000.pt'


def processes():
    return subprocess.check_output(['ps','-eo','pid,args'],text=True)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',default='assets/decoder_only_20261026')
    ap.add_argument('--resident-pid',type=int,default=0,
                    help='Explicit audited idle resident CUDA context, never another trainer')
    ap.add_argument('--resident-max-mib',type=int,default=0)
    ap.add_argument('--start',choices=['T2','V0','V1'],default='T2',help='Explicit completed-stage continuation')
    a=ap.parse_args(); os.chdir(ROOT)
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    lock=open('/tmp/dit_gpu0.lock','a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock.seek(0); lock.truncate(); lock.write(str(os.getpid())); lock.flush()
    # Do not race teval or silently stop another job. Existing eval exits itself.
    while any('tools.stroke_interaction.f4_gt_vs_gen ' in line for line in processes().splitlines()):
        (out/'status.json').write_text(json.dumps(dict(state='waiting_for_existing_T0_T1_evaluation',pid=os.getpid())))
        time.sleep(15)
    stale=[line for line in processes().splitlines() if 'python' in line and 'tools.three_day.' in line]
    if stale: raise RuntimeError('Old scheduler still alive: '+str(stale))

    def run(label,args):
        (out/'status.json').write_text(json.dumps(dict(state='checking_gpu_owner',arm=label,pid=os.getpid())))
        # Check actual device occupancy at each serial boundary, not only locks
        # (legacy launchers do not honor our lock). No automatic GPU-owner kill.
        owners=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory',
                                         '--format=csv,noheader,nounits'],text=True).strip()
        for line in owners.splitlines():
            pid,mib=[int(v.strip()) for v in line.split(',')]
            if pid!=a.resident_pid or mib>a.resident_max_mib:
                raise RuntimeError('GPU owned outside serial queue: '+owners)
        if owners:
            util=int(subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu',
                                               '--format=csv,noheader,nounits'],text=True).strip())
            if util>5: raise RuntimeError('Audited resident context became active')
            print('AUDITED_IDLE_RESIDENT',owners,flush=True)
        (out/'status.json').write_text(json.dumps(dict(state='running',arm=label,pid=os.getpid(),command=args)))
        print('START',label,args,flush=True)
        with open(out/f'{label}.log','a') as log:
            child=subprocess.Popen([sys.executable,'-u',*args],stdout=log,stderr=subprocess.STDOUT)
            rc=child.wait()
        if rc:
            (out/'status.json').write_text(json.dumps(dict(state='failed',arm=label,returncode=rc,pid=os.getpid())))
            raise RuntimeError(f'{label} failed rc={rc}; queue stopped')
        print('DONE',label,flush=True)

    if a.start=='T2':
        run('T2_smoke',['-m','tools.style_24h4090.smoke_t2'])
        run('T2',['-m','tools.style_24h4090.run_ablation','--arm','T2','--steps','5000',
                  '--batch','256','--rank-tau','0.5','--out-root',str(out/'fm')])
    else:
        assert list((out/'fm'/'T2').glob('*/checkpoints/0005000.pt')), 'T2 not completed'
    for arm in ('V0','V1'):
        if a.start=='V1' and arm=='V0': continue
        run(arm,['-m','tools.vae.decoder_only.train','--p2',P2,'--arm',arm,'--output',str(out/arm)])
        run(arm+'_eval',['-m','tools.vae.decoder_only.evaluate','--p2',P2,
                         '--vae',str(out/arm/'0002500'),'--out',str(out/(arm+'_eval'))])
    (out/'status.json').write_text(json.dumps(dict(state='complete',pid=os.getpid())))


if __name__=='__main__': main()
