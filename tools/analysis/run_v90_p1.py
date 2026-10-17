"""Finite isolated P1 A/B/control queue. No v90 base training is authorized here."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(2**20), b''): h.update(block)
    return h.hexdigest()


def prepare(root, work):
    work.mkdir(parents=True, exist_ok=False)
    for name in ('src', 'tools', 'tests'):
        shutil.copytree(root/name, work/name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('data', 'exp-std', 'pretrained_models', '_fonts'):
        if (root/name).exists(): (work/name).symlink_to(root/name, target_is_directory=True)
    (work/'assets').mkdir(); (work/'logs').mkdir(); (work/'scratch').mkdir()
    for path in (root/'assets').iterdir():
        if path.name != 'results': (work/'assets'/path.name).symlink_to(path, target_is_directory=path.is_dir())
    (work/'assets/results').mkdir()
    base = json.loads((work/'src/train/configs/v84A_std_cond.json').read_text())
    base.update(max_steps=5000, epochs=5000, lr=1e-5, warmup_steps=500,
        warm_start_from=str(root/'assets/runs/v84A_std_cond/20261013-072215-v84A-std-cond/checkpoints/0200000.pt'),
        ckpt_every=1000, epoch_steps=1000, ckpt_keep=0, global_seed=90,
        global_batch_size=224, num_workers=4, in_mem_eval_batch=11, in_mem_eval_vae_batch=8,
        in_mem_eval_sets='eval200fix:exp-std/csv/eval200_fixed.csv:187',
        in_mem_eval_poster=False, in_mem_eval_lpips=True, log_every=50,
        strong_style_mode='off', strong_style_probe_every=50, strong_style_max_weight=10.,
        rank_t_min=.4, rank_t_max=.8, rank_margin=.01, rank_tau=.1,
        w_critic_writer=0., w_critic_away=0., w_style_rank=0., freeze_triple_tables=True)
    # 11 divides187: no padded-tail mutation in historical sampling code.
    for arm, pair, ratio in [('A', .25, 0.), ('B', .25, .075), ('Control', 0., 0.)]:
        config = dict(base, experiment_name=f'v90-P1-{arm}', results_dir=f'assets/results/v90_P1_{arm}',
                      pair_fraction=pair, rank_gradient_ratio=ratio)
        (work/f'{arm}.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    smoke = dict(config, experiment_name='v90-P1-smoke', results_dir='assets/results/v90_P1_smoke',
        pair_fraction=.25, rank_gradient_ratio=.075, max_steps=4, epochs=4, ckpt_every=4, epoch_steps=4,
        in_mem_eval=True, in_mem_eval_sets='probe:exp-std/csv/eval200_fixed.csv:11', log_every=1, strong_style_probe_every=1)
    (work/'smoke.json').write_text(json.dumps(smoke, ensure_ascii=False, indent=2), encoding='utf-8')
    provenance = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'warm_start_sha256': sha(Path(base['warm_start_from'])), 'weights': 'ema', 'steps_per_arm': 5000,
        'batch': 224, 'pair_rows_per_batch':56, 'B_gradient_ratio':.075,
        'A_B_differ_only': ['experiment_name', 'results_dir', 'rank_gradient_ratio'],
        'p0_gate_override': 'Explicit user instruction: stop D3, complete P1 A/B; P2 remains disabled.',
        'source_sha256': {str(p.relative_to(work)):sha(p) for p in (work/'src').rglob('*.py')}}
    assert provenance['warm_start_sha256']=='6902bd9d92577e9c78c531ae19dbff2f3c43f74c41834472c69a71e611ec968b'
    aa=json.loads((work/'A.json').read_text());bb=json.loads((work/'B.json').read_text())
    assert {k for k in aa if aa[k]!=bb[k]}==set(provenance['A_B_differ_only'])
    (work/'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    # D3 has been explicitly stopped; preserve its files and mark the queue honestly.
    old=root/'scratch/writer_geometry_d2d3_20261016_01/assets/job_checkpoint.json'
    state=json.loads(old.read_text());state.update(status='stopped_by_user',cancelled_for='v90 P1 A/B',
        stopped_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    old.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
    print('P1_PREPARED',work,flush=True)


def run(root, work):
    os.chdir(work)
    lock=open(root/'logs/strong_style_queue.lock','a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=dict(os.environ, OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',PYTHONPATH=str(work))
    state={'pid':os.getpid(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'status':'running','work':str(work),'completed':[],'no_P2':True}
    path=work/'job_checkpoint.json'
    def save():path.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
    def stage(name,args):
        state.update(stage=name,command=[sys.executable,'-u',*args]);save();print('P1_STAGE',name,flush=True)
        with open(work/'logs'/f'{name}.log','w') as log:
            proc=subprocess.Popen([sys.executable,'-u',*args],env=env,stdout=log,stderr=subprocess.STDOUT)
            state['child_pid']=proc.pid;save();code=proc.wait()
        if code:
            state.update(status='failed',returncode=code);save();raise RuntimeError(f'{name} failed, see logs/{name}.log')
        state['completed'].append(name);save()
    stage('tests',['-m','pytest','-q','tests/test_pair_rank.py','tests/test_v90_p0.py'])
    stage('smoke',['-m','src.train.train','--config','smoke.json'])
    smoke=(work/'logs/smoke.log').read_text()
    if '[rank-fm]' not in smoke or 'Reached max_steps=4' not in smoke:
        raise RuntimeError('Smoke did not exercise rank and clean completion')
    for arm in ('A','B','Control'):
        stage('train_'+arm,['-m','src.train.train','--config',f'{arm}.json'])
        if 'Reached max_steps=5000' not in (work/f'logs/train_{arm}.log').read_text():
            raise RuntimeError('Training exited without 5k completion: '+arm)
    stage('evaluate',['tools/analysis/eval_v90_p1.py','--work',str(work),'--root',str(root)])
    state.update(status='complete',stage='complete');save();print('P1_COMPLETE',flush=True)


def repair_batch(work, batch):
    # No formal training has started: preserve the failed288 smoke, then lower ALL arms together.
    state=json.loads((work/'job_checkpoint.json').read_text())
    assert state['status']=='failed' and state['stage']=='smoke'
    assert not any((work/f'assets/results/v90_P1_{arm}').exists() for arm in ('A','B','Control'))
    old_batch=json.loads((work/'A.json').read_text())['global_batch_size']
    previous=work/f'attempt_batch{old_batch}';previous.mkdir(exist_ok=False)
    shutil.copy2(work/'logs/smoke.log',previous/'smoke.log')
    shutil.copy2(work/'job_checkpoint.json',previous/'job_checkpoint.json')
    shutil.copy2(work/'provenance.json',previous/'provenance.json')
    for arm in ('A','B','Control','smoke'):
        path=work/f'{arm}.json';config=json.loads(path.read_text());config['global_batch_size']=batch
        if arm=='smoke':
            config.update(in_mem_eval=True,in_mem_eval_sets='probe:exp-std/csv/eval200_fixed.csv:11')
        path.write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    path=work/'provenance.json';data=json.loads(path.read_text());data.update(batch=batch,pair_rows_per_batch=batch//4,
        preflight_revision=f'FM retained-graph gradient calibration OOM at288/256; reduce all arms identically to{batch}; no recomputation')
    path.write_text(json.dumps(data,indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--work',required=True)
    p.add_argument('--prepare',action='store_true');p.add_argument('--repair-batch',type=int,default=0);a=p.parse_args()
    root,work=Path(a.root).resolve(),Path(a.work).resolve()
    if a.prepare:prepare(root,work)
    elif a.repair_batch:repair_batch(work,a.repair_batch)
    else:run(root,work)
