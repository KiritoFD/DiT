"""Prepare the writer-table-only intervention and a matched frozen control."""
import hashlib
import json
import sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.train.cli import parse_args


def main():
    base=json.loads((ROOT/'src/train/configs/v85E_critic.json').read_text(encoding='utf-8'))
    ckpt=ROOT/base['warm_start_from']
    ck=torch.load(ckpt,map_location='cpu',weights_only=False,mmap=True)
    state=ck['ema']
    keys=[k for k in state if k.endswith('y_callig_embedder.embedding_table.weight')]
    assert len(keys)==1
    table=state[keys[0]]
    assert table.shape==(11,32)
    anchor=ROOT/'assets/v86A_v84A_writer_anchor.pt'
    torch.save({'writer_rows':table[:10].clone(),'source_checkpoint':str(ckpt),
                'source_sha256':hashlib.sha256(ckpt.read_bytes()).hexdigest()},anchor)
    for arm in ['frozen','table']:
        c=dict(base)
        c.update(experiment_name=f'v86A-writer-{arm}-20k',results_dir=f'assets/results/v86A_writer_{arm}',
                 max_steps=20000,epochs=20000,epoch_steps=5000,ckpt_every=5000,compile=True,
                 lr=1e-5,warmup_steps=1000,lr_schedule='cosine',min_lr_ratio=.1,
                 global_batch_size=192,global_seed=0,freeze_triple_tables=True,
                 triple_table_lr_scale=0.,w_critic_writer=0.,w_critic_away=0.,probe_style_grad_steps=0,
                 finetune_callig_table=arm=='table',callig_table_lr=2e-6,
                 callig_table_anchor_weight=.01,callig_table_anchor_file=str(anchor.relative_to(ROOT)))
        c['_comment']='Matched 20k continuation from v84A200k EMA; only the real writer table is unfrozen in table arm. Trunk/FM/REPA/STD and all optimizer schedule settings match. CFG null fixed, script/char tables frozen, writer group weight_decay=0.'
        p=ROOT/'src/train/configs'/f'v86A_writer_{arm}_20k.json'
        p.write_text(json.dumps(c,ensure_ascii=False,indent=2),encoding='utf-8')
        a=parse_args(['--config',str(p)])
        assert a.epoch_steps==a.ckpt_every==5000 and a.max_steps==20000
        print(p,arm,flush=True)


if __name__=='__main__':main()
