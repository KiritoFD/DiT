"""Exp F: FM vs FM+weak-E-G 2k short forks from S5@85k on 48.

Both forks: identical init (S5@85k EMA), identical data (Top10 p2_train CSV
order), LR 3e-6 flat, batch 512, 2000 steps. Fork B adds the E-G ranking
objective (every 4 steps, micro 8, auto-calibrated to 2% of FM grad norm).
E-G scores of fork B are diagnostics only.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',default='assets/p125_20261030/forks48')
    p.add_argument('--ckpt',default='assets/style_focus_85k_20261029/48_v3/runs/S5/checkpoints/0085000.pt')
    p.add_argument('--refs',default='assets/p125_20261030/eg_refs.npz')
    p.add_argument('--steps',type=int,default=2000)
    a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    base=json.loads(Path('assets/style_focus_85k_20261029/48_v3/specs/S5.json').read_text())
    cfg=base['config']
    cfg.update(max_steps=a.steps,lr_decay_steps=a.steps,lr=3e-6,lr_backbone=3e-6,
        lr_style_repr=3e-6,lr_style_inj=3e-6,lr_writer_table=3e-6,warmup=100,
        drop_writer=0.1,pair_fraction=0.3,ckpt_every=a.steps,eval_every=a.steps,
        null_writer_rows=False,char_remap_json='assets/triple_tables_best_minimal/char_remap.json')
    contract=dict(p125_fork='v1',global_batch=512,seed=90,compile=True,gradient_checkpointing=False,
        source_checkpoint_sha256=None)
    from tools.style_matrix.runtime.common import save
    forks={'FMA':dict(config=copy_cfg(cfg)),
           'FMB':dict(config=copy_cfg(cfg))}
    plans=[]
    for name,f in forks.items():
        spec=dict(arm=name,machine='48',kind='joint' if name=='FMB' else 'style',
            initialization='stage_transfer',source=a.ckpt,config=f['config'],
            milestones=[1000,a.steps],
            identity=dict(arm=name,contract=contract,config=f['config']))
        if name=='FMB':
            spec['continues_identity']=spec['identity']
            spec['joint']=dict(arm='EG',refs_npz=a.refs,every=4,image_micro=8,target_ratio=0.02)
        path=out/'specs'/f'{name}.json';save(path,spec)
        job=dict(arm=name,spec=str(path),stop=a.steps,previous_stop=0)
        plan=dict(machine='48',root=str(out/'execution'),ready=str(out/'ready.json'),
            contract=contract,wall_hours=2.5,estimated_upper_hours=0,promotion_limits={},
            stages={'1':[job]},
            evaluations=[dict(name='heldout187',csv='exp-std/csv/eval200_fixed.csv',writer_space='raw')])
        save(out/f'plan_{name}.json',plan)
        plans.append(str(out/f'plan_{name}.json'))
    print(json.dumps(dict(plans=plans,note='launch each with runtime.serial --execute'),indent=1))


def copy_cfg(c):
    import copy
    return copy.deepcopy(c)


if __name__=='__main__':main()
