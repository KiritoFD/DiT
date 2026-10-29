"""Matched 50k->85k continuation for S0/S4/S5; no promotion of the old matrix."""
import argparse
import copy
from pathlib import Path
from tools.style_matrix.generation2.plan import read,continuation,protocols,seal
from tools.style_matrix.runtime.common import save,sha
from tools.style_matrix.runtime.serial import verify_assets


def immutable(path,value):
    if path.exists():
        if read(path)!=value:raise ValueError('Immutable focused plan changed: '+str(path))
    else:save(path,value)


def main():
    p=argparse.ArgumentParser();p.add_argument('--machine',choices=['4090','48'],required=True)
    p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
    old=Path('assets/style_matrix_20261027/matrix64')/a.machine
    arms=['S0','S4'] if a.machine=='4090' else ['S5'];jobs=[];records=[]
    for arm in arms:
        base=read(old/'specs'/f'{arm}.json');cfg=base['config']
        assert cfg['batch']==512 and cfg['compile'] and cfg['lr']==6e-5
        assert not base.get('joint') and not base.get('velocity_pair_weight')
        checkpoint=old/'execution/runs'/arm/'checkpoints/0050000.pt'
        done=read(str(checkpoint)+'.done')
        if done['step']!=50000 or checkpoint.stat().st_size!=done['size']:raise ValueError('Source checkpoint incomplete')
        if read(checkpoint.parents[1]/'identity.json')!=base['identity']:raise ValueError('Source run identity changed')
        spec=continuation(base,arm,50000,85000)
        spec['config']['ckpt_every']=5000
        spec['config']['eval_every']=5000
        spec['milestones']=list(range(55000,85001,5000))
        spec['identity']=dict(study='focused_S0_S4_S5_85k',arm=arm,config=spec['config'],
            parent=base['identity'],restart=spec['schedule_restart'])
        path=root/'specs'/f'{arm}.json';immutable(path,spec)
        jobs.append(dict(arm=arm,spec=str(path),spec_sha256=sha(path),stop=85000,
            previous_stop=50000,fork_from=str(checkpoint)))
        records.append(dict(arm=arm,checkpoint=str(checkpoint),size=done['size'],source_step=50000,
            source_identity=base['identity'],train_csv_sha256=sha(cfg['data_csv']),
            batch=cfg['batch'],microbatch=cfg['microbatch'],lr=cfg['lr'],seed=cfg['seed']))
    evaluation=protocols(root,base,'S')
    evaluation=[dict(name='original_val100',csv=base['config']['eval_csv'],writer_space='raw')]+evaluation
    seal(root,a.machine);ready=read(root/'ready.json')
    # Reclassify all inherited assets by the current large-asset sealing policy.
    for item in ready['files']:
        file=Path(item['path'])
        if not file.is_file():raise FileNotFoundError(file)
        if file.stat().st_size>1<<20:
            item.update(size=file.stat().st_size,sha256=None)
    immutable(root/'inputs.json',dict(sources=records,effective_batch=512,added_updates=35000,
        added_exposure_per_model=512*35000,new_generator_architecture=False))
    save(root/'ready.json',ready)
    plan=dict(machine=a.machine,root=str(root),ready=str(root/'ready.json'),contract=ready['contract'],
        wall_hours=24,estimated_upper_hours=14 if a.machine=='4090' else 7,
        stages={'1':jobs},evaluations=evaluation)
    immutable(root/'plan.json',plan);verify_assets(plan)
    print('READY',a.machine,arms,'50k->85k',dict(warmup=200,peak=3e-5,end=3e-6,
        effective_batch=512,exposure_per_model=17920000),flush=True)


if __name__=='__main__':main()
