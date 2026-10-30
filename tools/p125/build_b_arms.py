"""Exp B: three-arm P1 ablation from the same P0@50k (on 48).
B0 = original 70/30 mix; B1 = 100% Top10 full FT; B2 = 100% Top10,
writer-table+project only (train_scope). All 5k, same batch/seeds.
Runs on 48 with the transferred P0@50000 checkpoint.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
sys.path.insert(0,'.')
from tools.style_matrix.runtime.common import save

OUT=Path("assets/p125_20261030/b_arms")


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--p1-spec',default='assets/p125_20261030/specs/P1.json')
    p.add_argument('--p0-ckpt',default='assets/p125_20261030/execution/runs/P0/checkpoints/0050000.pt')
    a=p.parse_args()
    p1=json.loads(Path(a.p1_spec).read_text())
    base=copy.deepcopy(p1['config'])
    contract=dict(p125_b='v1',global_batch=512,seed=90,compile=True,gradient_checkpointing=False,
        num_writers=base['num_writers'],source_checkpoint=str(a.p0_ckpt))
    out=OUT;(out/'specs').mkdir(parents=True,exist_ok=True)
    arms={}
    for name,data_csv,scope in (('B0',base['data_csv'],'full'),
                                ('B1','assets/p125_20261030/stage_p2.csv','full'),
                                ('B2','assets/p125_20261030/stage_p2.csv','writer_table_only')):
        c=copy.deepcopy(base)
        c['data_csv']=data_csv
        c['max_steps']=5000;c['lr_decay_steps']=5000;c['warmup']=200
        c['drop_writer']=0.25 if name=='B0' else 0.1
        if scope=='writer_table_only':c['train_scope']='writer_table_only'
        spec=dict(arm=name,machine='48',kind='style',initialization='stage_transfer',
            source=a.p0_ckpt,config=c,milestones=[1000,2500,5000],
            identity=dict(arm=name,contract=contract,config=c))
        path=out/'specs'/f'{name}.json';save(path,spec)
        job=dict(arm=name,spec=str(path),stop=5000,previous_stop=0)
        plan=dict(machine='48',root=str(out/'execution'),ready=str(out/'ready.json'),
            contract=contract,wall_hours=1.6,estimated_upper_hours=0,promotion_limits={},
            stages={'1':[job]},
            evaluations=[dict(name='heldout',csv=base['eval_csv'],writer_space='top10-row')])
        save(out/f'plan_{name}.json',plan)
        arms[name]=dict(spec=str(path),data_rows=data_csv)
    print(json.dumps(dict(arms=arms),indent=1))


if __name__=='__main__':main()
