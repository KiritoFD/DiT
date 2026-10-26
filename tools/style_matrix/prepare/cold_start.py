"""Seal and launchable-plan preparation for the revised scratch S study."""
import argparse
import copy
import csv
import json
from pathlib import Path
from ..runtime.common import save,sha


ARMS={'S0':(0,0),'S1':(0,1),'S2':(0,2),'S3':(1,2),
      'S4':(2,1),'S5':(3,1),'S6':(0,3),'S7':(3,3)}
HOSTS={'4090':['S0','S1','S4','S6'],'48':['S2','S3','S5','S7']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--base-spec',required=True)
    p.add_argument('--machine',choices=HOSTS,required=True);p.add_argument('--out',required=True)
    p.add_argument('--microbatch',type=int,default=512);p.add_argument('--steps',type=int,default=50000)
    a=p.parse_args();out=Path(a.out)
    if out.exists():raise FileExistsError('Use a new sealed plan directory')
    base=json.loads(Path(a.base_spec).read_text())['config']
    rows=list(csv.DictReader(open(base['data_csv'],encoding='utf-8-sig')))
    ids={int(r['calligrapher_id']) for r in rows}
    if ids!=set(range(10)):raise ValueError('Expected canonical Top10 writer rows')
    save(out/'callig_remap.json',{str(i):i for i in range(10)})
    base.update(num_writers=10,callig_remap_json=str(out/'callig_remap.json'),compile=True,
        microbatch=a.microbatch,batch=512,max_steps=a.steps,lr_decay_steps=a.steps,
        lr=6e-5,weight_decay=.02,warmup=1500,min_lr_ratio=.1,ema_decay=.999,ema_interval=1,
        initialize_from='',requires_warm_start=False,ckpt_every=5000,eval_every=5000,
        checkpoint_steps=[],stop_at_step=0,std_shape_step_offset=0)
    contract=dict(initialization='scratch',global_batch=512,seed=base['seed'],compile=True,
        gradient_checkpointing=False,train_csv_sha256=sha(base['data_csv']),
        eval_csv_sha256=sha(base['eval_csv']),steps=a.steps)
    jobs=[]
    for arm in HOSTS[a.machine]:
        config=copy.deepcopy(base);r,i=ARMS[arm]
        config.update(style_matrix_repr=r,style_matrix_injection=i,style_matrix_vector_dim=32)
        identity=dict(arm=arm,contract=contract,config=config)
        spec=dict(arm=arm,machine=a.machine,kind='style',initialization='scratch',source=None,
            config=config,milestones=sorted(set([1000,5000,10000,25000,a.steps])),identity=identity)
        path=out/'specs'/f'{arm}.json';save(path,spec)
        jobs.append(dict(arm=arm,spec=str(path),spec_sha256=sha(path),stop=a.steps,previous_stop=0))
    plan=dict(machine=a.machine,root=str(out/'execution'),ready=str(out/'ready.json'),contract=contract,
        wall_hours=64.,estimated_upper_hours=len(jobs)*a.steps/1.7/3600+4,
        rate_basis='Historical Deep114 compiled ~1.7 step/s; estimate, not new-interface measurement',
        stages={'1':jobs},evaluations=[dict(name='heldout',csv=base['eval_csv'],writer_space='top10-row')])
    save(out/'plan.json',plan)
    paths={str(out/'callig_remap.json'),*[j['spec'] for j in jobs]}
    for key in ('data_csv','eval_csv','std_shape_cache','char_remap_json','callig_remap_json','font_remap_json'):
        paths.add(base[key])
    for key in ('latent_shards_dir','std_soft_root','vae_path'):
        files=[str(f) for f in Path(base[key]).rglob('*') if f.is_file()]
        if not files:raise FileNotFoundError(base[key])
        paths.update(files)
    for root in ('src','tools/style_matrix'):
        paths.update(str(f) for f in Path(root).rglob('*.py'))
    save(out/'ready.json',dict(contract=contract,files=[dict(path=f,sha256=sha(f)) for f in sorted(paths)],
        no_training_started=True))
    print(json.dumps(dict(plan=str(out/'plan.json'),queue=HOSTS[a.machine],
        steps=a.steps,microbatch=a.microbatch,estimated_hours=plan['estimated_upper_hours']),indent=2))


if __name__=='__main__':main()
