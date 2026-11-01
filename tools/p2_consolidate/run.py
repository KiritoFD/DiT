"""Two model-only P2 EMA transfers, identical training apart from STD severity."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch
from tools.p2_audit.common import save,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True)
    p.add_argument('--out',default='assets/p2_std_consolidation_20261025')
    p.add_argument('--steps',type=int,default=5000);a=p.parse_args()
    root=Path(a.out);root.mkdir(parents=True,exist_ok=True)
    if (root/'status.json').exists():raise FileExistsError('Use a new output for a new queue; no implicit reruns')
    from src.unet.io import checkpoint_config,Config
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);source=checkpoint_config(ck)
    source_step=int(ck['train_steps']);del ck
    if source.std_shape_aug!='strong':raise ValueError('Expected P2 strong-augmentation source')
    base=asdict(source)
    base.update(initialize_from=a.checkpoint,initialize_state='ema',max_steps=a.steps,stop_at_step=a.steps,
                lr_decay_steps=a.steps,lr=5e-6,writer_cond_lr=2e-5,warmup=100,batch=512,
                ckpt_every=a.steps//2,eval_every=a.steps//2,eval_before_training=False,
                std_shape_step_offset=source.std_shape_step_offset+source_step,
                std_soft_eval_sigma=0.,ema_start_step=0)
    configs={}
    for arm,mode in [('A_strong','strong'),('B_mild','mild')]:
        configs[arm]=dict(base,std_shape_aug=mode,output=str(root/arm/'runs'))
        Config(**configs[arm]).validate();save(root/(arm+'.json'),configs[arm])
    differences={k for k in configs['A_strong'] if configs['A_strong'][k]!=configs['B_mild'][k]}
    if differences!={'std_shape_aug','output'}:raise ValueError(f'Confounded arms: {differences}')
    status=dict(pid=os.getpid(),state='running',started=time.strftime('%F %T'),source_sha256=sha(a.checkpoint),
                source_step=source_step,steps_per_arm=a.steps,phases={})
    save(root/'manifest.json',dict(configs=configs,source_sha256=status['source_sha256'],
        comparison='A vs B, NOT B initialized from A',optimizer_reset_both=True,ema_reset_both=True,
        expected_requested_tiers=[.20,.25,.40,.15],expected_mild_effective_tiers=[.20,.80,0,0],
        examples_per_arm=512*a.steps,paired_examples_per_batch=152,
        caveat='Same RNG streams and sample sequence, not bitwise equal numerical trajectories'))
    for arm in configs:
        cmd=[sys.executable,'-u','-m','src.unet.train','--config',str(root/(arm+'.json'))]
        status.update(current=arm,command=cmd);status['phases'][arm]={'state':'running','started':time.strftime('%F %T')}
        save(root/'status.json',status)
        with (root/(arm+'.log')).open('w') as log:
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            status.update(state='failed',returncode=result.returncode);save(root/'status.json',status)
            raise SystemExit(f'{arm} failed; later arms not launched')
        run=next((root/arm/'runs').glob('*-Unet-flow'))
        final=run/'checkpoints'/f'{a.steps:07d}.pt'
        if not final.exists():raise FileNotFoundError(final)
        transfer=json.loads((run/'stage_transfer.json').read_text())
        if not transfer['all_tensors_equal'] or transfer['sha256']!=status['source_sha256']:
            raise ValueError('Arm did not initialize exactly from the pinned source')
        status['phases'][arm].update(state='done',checkpoint=str(final),finished=time.strftime('%F %T'))
        save(root/'status.json',status)
    # Scoring reads existing inline evaluation PNGs; no second sampling run.
    from .evaluate import evaluate
    status['current']='evaluate';save(root/'status.json',status)
    try:evaluate(root,configs)
    except Exception as e:
        status.update(state='failed',error=repr(e));save(root/'status.json',status);raise
    status.update(state='done',finished=time.strftime('%F %T'));save(root/'status.json',status)


if __name__=='__main__':main()
