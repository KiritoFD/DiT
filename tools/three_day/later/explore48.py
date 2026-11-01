"""48-side Deep114 R0/R1 exploration; R2/R3 only with separately accepted signals."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
import torch
from tools.three_day.contracts import save,sha,checkpoint_rows
from tools.three_day.run import Queue


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
    queue=Queue(root/'R');source=Path('assets/results/unet_deep_sym4_114m_60m/20261019-011647-Unet-flow/checkpoints/0085000.pt')
    from src.unet.io import checkpoint_config,Config
    ck=torch.load(source,map_location='cpu',weights_only=False,mmap=True);c=checkpoint_config(ck)
    base=asdict(c);base.update(initialize_from=str(source),initialize_state='ema',batch=256,seed=90,
        max_steps=3000,stop_at_step=3000,lr_decay_steps=3000,lr=5e-6,writer_cond_lr=2e-5,
        ckpt_every=1500,eval_every=1500,warmup=100,ema_start_step=0,eval_before_training=False,
        std_shape_step_offset=c.std_shape_step_offset+int(ck['train_steps']),
        semantic_writer_rows={str(k):v for k,v in checkpoint_rows(c).items()},compile=False)
    gates=json.loads((root/'style_gates.json').read_text())
    review=json.loads((root/'visual_review.json').read_text())
    specs={'R0':dict(objective='fm',external_dropout=True)}
    if 'D2' in gates['global_candidates'] and review['D2']['allow_short_training']:
        specs['R1']=dict(objective='global_style',external_dropout=True,teacher='D2',weight=.02,
            teacher_checkpoint=str(root/'teachers/D2.pt'),reference_cache=str(root/'style_reference_D2.pt'),
            reference_csv_sha256=review['reference_csv_sha256'])
    for arm,spec in specs.items():
        path=root/'R'/arm/'study.json';save(path,spec)
        config=dict(base,output=str(root/'R'/arm/'runs'),study_spec=str(path));Config(**config).validate()
        p=root/'R'/arm/'config.json';save(p,config)
        queue.command(arm,[sys.executable,'-u','-m','src.unet.train','--config',str(p)])
        if len(list((root/'R'/arm/'runs').glob('*/checkpoints/0003000.pt.done')))!=1:raise FileNotFoundError(arm+' final missing')
    save(root/'R/gated_arms.json',dict(R2='held: local patch failed F1/F2; no generator training',
        R3='held: available DINO content representation failed GT/STD retrieval; content teacher/gradient gate not passed',
        remaining='R0/R1 contrast and T1/T2/T3/T4 still require final evaluation and explicit gates'))
    queue.status.update(state='R_training_complete');queue.flush()


if __name__=='__main__':main()
