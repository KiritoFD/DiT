"""Bounded full-training profiles from the same checkpoint and seed."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

parser=argparse.ArgumentParser()
parser.add_argument('--base',required=True)
parser.add_argument('--checkpoint',required=True)
parser.add_argument('--variants',nargs='+',default=['current','repa_sync','repa_async','fusion_on'])
args=parser.parse_args()
base=json.loads(Path(args.base).read_text())
env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',PYTHONPATH=str(Path.cwd()))
records=[]
variants=[('current',0.,False,'legacy'),('repa_sync',.03,False,'legacy'),
          ('repa_async',.03,True,'legacy'),('fusion_on',0.,False,'on'),
          ('guided',0.,False,'on')]
for name,weight,prefetch,pattern in variants:
    if name not in args.variants:continue
    config=dict(base,experiment_name='profile_'+name,results_dir='assets/profile_results/'+name,
                warm_start_from=args.checkpoint,resume_full=None,pretrained=None,
                max_steps=80,stop_at_step=80,epochs=80,ckpt_every=1000,epoch_steps=1000,
                global_seed=90,log_every=20,in_mem_eval=False,w_repa=weight,
                repa_cache_prefetch=prefetch,compile_pattern_matcher=pattern,
                profile_steps=10,profile_skip=60,std_shape_log_every=1000)
    config['repa_cache_dir']='data/dino_cache/top10_aug_v1' if weight else ''
    if name=='guided':
        config.update(compile_mode='max-autotune-no-cudagraphs',compile_fullgraph=True)
    path=Path('logs')/(name+'.json');path.write_text(json.dumps(config,indent=2))
    state=Path('profile_status.json')
    state.write_text(json.dumps(dict(status='running',variant=name,records=records),indent=2))
    with (Path('logs')/(name+'.log')).open('w') as log:
        child=subprocess.Popen([sys.executable,'-u','-m','src.train.train','--config',str(path)],
                               env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=child.wait(timeout=600)
        except subprocess.TimeoutExpired:
            import signal
            os.killpg(child.pid,signal.SIGTERM);child.wait();code=124
    records.append(dict(name=name,returncode=code))
state.write_text(json.dumps(dict(status='finished',records=records),indent=2))
