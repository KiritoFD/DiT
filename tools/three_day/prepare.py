"""Build the 2x2 experiment without duplicating the trainer or large caches."""
from collections import Counter
import json
from pathlib import Path
import shutil
from .contracts import TOP10,read,save,sha

BUNDLE=Path('assets/u100_pretrain_20261024')
OLD=Path('assets/u100_p1p2_20261025')


def prepare(root):
    from src.unet.io import Config
    root=Path(root);manifest=root/'manifest.json'
    if manifest.exists():return json.loads(manifest.read_text())
    root.mkdir(parents=True,exist_ok=True)
    tables=root/'tables';tables.mkdir()
    for name in ('char_remap.json','font_remap.json','font_table.npy','callig_table.npy'):
        shutil.copy2(BUNDLE/'tables'/name,tables/name)
    save(tables/'callig_remap.json',{str(i):i for i in range(260)})
    for name in ('p1_train.csv','p2_train.csv'):shutil.copy2(OLD/name,root/name)
    p0s=list((BUNDLE/'runs_100k').glob('*/checkpoints/0075000.pt'))
    if len(p0s)!=1:raise ValueError('P0@75k must resolve uniquely')
    p0=str(p0s[0]);base=json.loads((OLD/'p2.json').read_text())
    # Uniform drop probability across both stages AND all arms: isolate only the two factors.
    base.update(initialize_from=p0,initialize_state='ema',batch=512,seed=90,lr=1e-5,
        writer_cond_lr=5e-5,drop_writer=.10,warmup=200,ema_start_step=0,
        std_soft_eval_sigma=0.,eval_before_training=False,null_writer_rows=True,
        semantic_writer_rows={str(k):v for k,v in TOP10.items()},
        condition_tables_root=str(tables),pair_fraction=0.,ckpt_every=5000,eval_every=5000)
    for name in ('callig','char','font'):base[name+'_remap_json']=str(tables/(name+'_remap.json'))
    for name in ('p1_train.csv','p2_train.csv'):
        rows=read(root/name)
        if any(int(r['calligrapher_id']) not in set(range(10))|{259} for r in rows):
            raise ValueError('Unexpected writer rows in frozen training CSV')
    arms={}
    for arm,mixed,paired in [('A0',False,False),('A3',True,True),('A1',True,False),('A2',False,True)]:
        phases=[]
        for stage,steps in [('p1',10000),('p2',20000)]:
            config=dict(base,max_steps=steps,stop_at_step=steps,lr_decay_steps=steps,
                data_csv=str(root/('p1_train.csv' if mixed and stage=='p1' else 'p2_train.csv')),
                output=str(root/arm/stage/'runs'),pair_fraction=.3 if paired and stage=='p2' else 0.,
                std_shape_step_offset=0 if stage=='p1' else 10000)
            Config(**config).validate();path=root/arm/(stage+'.json');save(path,config)
            spec=root/arm/(stage+'_study.json');save(spec,dict(objective='fm',arm=arm,stage=stage))
            config['study_spec']=str(spec);save(path,config)
            phases.append(str(path))
        arms[arm]={'mixed':mixed,'paired':paired,'configs':phases,'steps':[10000,20000],
                   'exposure':15360000,'pair_endpoints_per_batch':152 if paired else 0}
    inputs=[root/'p1_train.csv',root/'p2_train.csv',BUNDLE/'validation_val100.csv',
            BUNDLE/'pretrain_100k.json',Path(base['std_soft_root'])/'index.csv',*tables.glob('*')]
    manifest_data=dict(source=p0,source_sha256=sha(p0),arms=arms,
        input_hashes={str(p):sha(p) for p in inputs},
        semantic_writer_rows=base['semantic_writer_rows'],
        common=dict(batch=512,backbone_lr=1e-5,condition_lr=5e-5,drop_writer=.10,seed=90,
            freeze_writer_table=base['freeze_writer_table'],reset_optimizer_and_ema_each_stage=True,
            std_augmentation_clock='continuous 0..30k across the two stages',sigma_eval=0.,
            test200_selection_forbidden=True),
        data={name:dict(n=len(read(root/name)),writer_counts=dict(Counter(r['calligrapher_id'] for r in read(root/name))))
              for name in ('p1_train.csv','p2_train.csv')},
        notes=['Mixed arm reuses exactly the prior P1 frozen resampled CSV; actual exposure counts logged separately.',
               'Pair sampler changes endpoint marginals as well as covariance; no extra contrastive/ranking loss.',
               'Inherited table trainability from P2, identical across all arms; no new independent-table experiment.'])
    save(manifest,manifest_data);return manifest_data
