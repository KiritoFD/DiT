"""Prepare the Deep85 decoder study with explicit current/legacy arm mapping."""
import argparse
import copy
import csv
from dataclasses import fields
import json
from pathlib import Path
from src.unet.io import Config
from ..runtime.common import save,sha


def seal(out):
    plan=json.loads((out/'plan.json').read_text());paths=set()
    calibration=out/'calibration.json';cal=json.loads(calibration.read_text())
    if not cal['paired_gt_checked']:raise ValueError('Real GT/latent pairing has not passed')
    if cal['source_checkpoint_sha256']!=plan['contract']['source_sha256']:
        raise ValueError('Calibration used another generator')
    paths.add(str(calibration))
    for job in plan['stages']['1']:
        spec=json.loads(Path(job['spec']).read_text());cfg=spec['config']
        if sha(cfg['data_csv'])!=cal['train_csv_sha256']:raise ValueError('Calibration data changed')
        paths.update((job['spec'],spec['source']))
        for key in ('data_csv','eval_csv','std_shape_cache','char_remap_json','callig_remap_json','font_remap_json'):
            paths.add(cfg[key])
        for key in ('latent_shards_dir','std_soft_root','vae_path'):
            files=[str(p) for p in Path(cfg[key]).rglob('*') if p.is_file()]
            if not files:raise FileNotFoundError(cfg[key])
            paths.update(files)
    paths.update(r['image_path'] for r in csv.DictReader(open(cfg['data_csv'],encoding='utf-8-sig')))
    for root in ('src','tools/style_matrix'):
        paths.update(str(p) for p in Path(root).rglob('*.py'))
    if (out/'ready.json').exists():raise FileExistsError('Already sealed')
    save(out/'ready.json',dict(contract=plan['contract'],files=[dict(path=p,sha256=sha(p)) for p in sorted(paths)],
        paired_gt_verified=True))
    print('SEALED',out,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-run',required=True);p.add_argument('--out',required=True)
    p.add_argument('--steps',type=int,default=5000);p.add_argument('--seal',action='store_true')
    a=p.parse_args();out=Path(a.out)
    if a.seal:return seal(out)
    if out.exists():raise FileExistsError('Use a new immutable plan directory')
    source_run=Path(a.source_run);source=source_run/'checkpoints/0085000.pt'
    raw=json.loads((source_run/'resolved_config.json').read_text());allowed={f.name for f in fields(Config)}
    config={k:v for k,v in raw.items() if k in allowed}
    with open(config['data_csv'],encoding='utf-8-sig') as f:
        reader=csv.DictReader(f);names=reader.fieldnames;rows=list(reader)
    clean=[r for r in rows if not (r.get('aug') or '').strip()]
    if not clean or len({r['img_id'] for r in clean})!=len(clean):raise ValueError('Invalid paired GT IDs')
    out.mkdir(parents=True);train=out/'train_unaugmented.csv'
    with train.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=names);writer.writeheader();writer.writerows(clean)
    config.update(data_csv=str(train),compile=True,batch=512,microbatch=512,
        max_steps=a.steps,lr_decay_steps=a.steps,warmup=300,lr=2e-5,weight_decay=.01,
        freeze_writer_table=False,seed=90,pair_fraction=.3,style_matrix_repr=-1,
        std_shape_step_offset=85000,ema_decay=.999,ema_interval=1,ckpt_every=2500,eval_every=2500,
        checkpoint_steps=[],initialize_from='',stop_at_step=0)
    config=vars(Config(**config))
    contract=dict(source_sha256=sha(source),source_step=85000,initialization='Deep85_EMA',
        global_batch=512,gradient_checkpointing=False,train_csv_sha256=sha(train),
        eval_csv_sha256=sha(config['eval_csv']),steps=a.steps,seed=90,
        external_to_internal={'J0':'J0','J1':'J1','J2':'J3'})
    jobs=[]
    for arm,internal in contract['external_to_internal'].items():
        cfg=copy.deepcopy(config)
        spec=dict(arm=arm,machine='48',kind='joint',initialization='Deep85_EMA',source=str(source),
            config=cfg,milestones=[1000,2500,a.steps],
            joint=dict(arm=internal,image_micro=8,image_batch=16,calibration=str(out/'calibration.json')),
            identity=dict(arm=arm,internal_objective=internal,contract=contract,config=cfg))
        path=out/'specs'/f'{arm}.json';save(path,spec)
        jobs.append(dict(arm=arm,spec=str(path),spec_sha256=sha(path),stop=a.steps,previous_stop=0))
    plan=dict(machine='48',root=str(out/'execution'),ready=str(out/'ready.json'),contract=contract,
        wall_hours=24,estimated_upper_hours=8,stages={'1':jobs},
        evaluations=[dict(name='heldout187',csv=config['eval_csv'],writer_space='raw')])
    save(out/'plan.json',plan)
    save(out/'data_selection.json',dict(original_rows=len(rows),unaugmented_rows=len(clean),
        reason='GT image supervision requires the exact unaugmented latent target; common to J0/J1/J2',
        writer_mapping='Original Deep85 raw semantic writer IDs, original remap retained',
        original_csv=raw['data_csv'],original_csv_sha256=sha(raw['data_csv'])))
    print(json.dumps(dict(plan=str(out/'plan.json'),queue=['J0','J1','J2'],train_rows=len(clean),
        source=str(source),source_sha256=contract['source_sha256']),indent=2),flush=True)


if __name__=='__main__':main()
