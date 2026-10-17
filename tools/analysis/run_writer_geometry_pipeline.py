"""Finite serial D2/D3 training -> heldout -> shared multiseed gradient audit."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--work',required=True);a=p.parse_args();root=Path(a.root);work=Path(a.work);os.chdir(work)
    python=sys.executable;started=time.time();checkpoint=work/'assets/job_checkpoint.json'
    lock=open(root/'logs/strong_style_queue.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state={'task':'D2/D3 geometry/brush controlled ablation','pid':os.getpid(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'work':str(work),'configuration':{'batch':2048,'steps_each':10000,'seed':89,'lr':1e-4,'cross_weight':.5,'D2_geometry_weight':0.,'D3_geometry_weight':.01,
            'checkpoint_steps':[1000,2000,3000,4000,5000,6000,8000,10000],'gradient_bank':'48 distinct heldout characters x 3 seeds; same bank for D1/D2/D3',
            'negative_protocol':'train-only writer prototypes, query character excluded; NOT heldout GT hard negatives'},
        'completed_stages':[],'no_dit_training':True}
    def save():
        state['elapsed_seconds']=time.time()-started;checkpoint.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
    def run(stage,args):
        state['status']='running';state['stage']=stage;state['current_command']=[python,'-u',*args];save();print('PIPELINE_STAGE',stage,flush=True)
        try:subprocess.run([python,'-u',*args],check=True)
        except subprocess.CalledProcessError as e:
            state['status']='failed';state['returncode']=e.returncode;save();raise
        state['completed_stages'].append(stage);save()
    run('tests',['-m','pytest','-q','tests/test_writer_style_geometry.py','tests/test_writer_style_latent.py','tests/test_latent_style_gradient.py','tests/test_geometry_audit_protocol.py'])
    for arm,weight in (('D2','0'),('D3','0.01')):
        run('train_'+arm,['tools/train_writer_style_pixel.py','--domain','latent','--architecture','geometry','--arm',arm,'--geometry-weight',weight,
            '--data-root','assets/data','--sdf-cache','assets/sdf32','--out',f'assets/{arm}','--batch','2048','--steps','10000','--eval-every','500'])
    for arm in ('D2','D3'):run('accept_'+arm,['tools/analysis/accept_writer_style_pixel.py','--data-root','assets/data','--run',f'assets/{arm}'])
    run('compare_real_gt',['tools/analysis/compare_writer_geometry.py','--root',str(root),'--work',str(work)])
    bank=work/'assets/gradient_bank_48x3'
    run('sample_48x3',['tools/analysis/prepare_style_gradient_bank.py','--root',str(root),'--out',str(bank)])
    backbone=root/'assets/latent_d1_prototype_gradient_audit_20261016/torch_cache/checkpoints/alexnet-owt-7be5be79.pth'
    for arm in ('D1','D2','D3'):
        encoder=root/'scratch/writer_latent_d0d1_20261016_01/assets/D1' if arm=='D1' else work/f'assets/{arm}'
        run('gradient_'+arm,['tools/analysis/audit_latent_d1_gradient.py','--root',str(root),'--out',str(work/f'assets/gradient_{arm}'),
            '--latent-run',str(encoder),'--generation-bank',str(bank),'--step-sizes','0.01','--negative-protocol','prototype','--roundtrip',
            '--reference-cache',str(work/'assets/gradient_judge_train_cache'),'--lpips-backbone',str(backbone)])
    run('compare_gradients',['tools/analysis/compare_geometry_gradients.py','--work',str(work)])
    state['status']='complete';state['stage']='complete';save();print('WRITER_GEOMETRY_PIPELINE_COMPLETE',flush=True)


if __name__=='__main__':main()
