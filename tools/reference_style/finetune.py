"""Top10 reference-only full fine-tuning from pretraining EMA; no Writer-ID branch."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from .prepare import read,write
from src.unet.io import Config


def extend(args):
    """Preserve raw/EMA/Adam/RNG; explicitly extend the existing cosine horizon."""
    checkpoint=Path(args.resume)
    base=json.loads((checkpoint.parent.parent/'resolved_config.json').read_text())
    old_horizon=base['lr_decay_steps'] or base['max_steps']
    if args.steps<=old_horizon:raise ValueError('Extension must increase the total horizon')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    base.update(max_steps=args.steps,stop_at_step=args.steps,lr_decay_steps=args.steps,
                output=str(out/'runs'),eval_before_training=False,initialize_from='')
    Config(**base).validate();(out/'config.json').write_text(json.dumps(base,indent=2))
    status=dict(phase='training',pid=os.getpid(),resume=str(checkpoint),
                old_horizon=old_horizon,new_horizon=args.steps,
                optimizer_reset=False,ema_reset=False,
                lr_policy='Rejoin existing base-LR cosine at the new total horizon; not identical old LR trajectory')
    def save():(out/'pipeline_status.json').write_text(json.dumps(status,indent=2))
    save()
    try:
        with (out/'training.log').open('w') as log:
            subprocess.run([sys.executable,'-u','-m','src.unet.train','--config',str(out/'config.json'),
                            '--resume',str(checkpoint)],stdout=log,stderr=subprocess.STDOUT,check=True)
        status['phase']='completed';save()
    except BaseException as error:
        status.update(phase='failed',error=repr(error));save();raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='assets/reference_top10_20261024')
    p.add_argument('--source',default='assets/generic48_kxl_top10_20261022')
    p.add_argument('--pretrain',default='assets/reference_kxl_20261023')
    p.add_argument('--checkpoint',default='assets/reference_kxl_20261023/runs/20261023-195603-Unet-flow/checkpoints/0025000.pt')
    p.add_argument('--steps',type=int,default=10000);p.add_argument('--resume',default='');a=p.parse_args()
    if a.resume:return extend(a)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);source=Path(a.source)
    status=dict(phase='preparing',pid=os.getpid())
    def save():(out/'pipeline_status.json').write_text(json.dumps(status,indent=2))
    def run(cmd,name):
        status['phase']=name;save()
        with (out/(name+'.log')).open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    save()
    try:
        train=[r for r in read(source/'top10_train.csv') if r['generic']=='0']
        validation=read(source/'top10_validation.csv');test=read(source/'top10_test.csv')
        names=sorted({r['calligrapher'] for r in train})
        if len(names)!=10:raise ValueError('Expected exactly 10 training writers')
        mapping={name:i for i,name in enumerate(names)}
        groups={}
        for r in train:groups.setdefault((r['calligrapher'],r['script']),set()).add(r['character'])
        if any(len(groups[r['calligrapher'],r['script']])<2 for r in train):
            raise ValueError('Top10 contains writer/script without crosschar references')
        for rows in (train,validation,test):
            for r in rows:
                r['style_writer_id']=str(mapping[r['calligrapher']]);r['style_writer_name']=r['calligrapher']
                r['style_writer_key']=r['calligrapher']
                if not (groups[r['calligrapher'],r['script']]-{r['character']}):
                    raise ValueError('No crosschar training reference for downstream target')
        train_hashes={r['pixel_sha256'] for r in train}
        if train_hashes & {r['pixel_sha256'] for r in validation+test}:raise ValueError('Heldout pixel overlap')
        write(out/'train.csv',train);write(out/'validation.csv',validation);write(out/'test.csv',test)
        (out/'data_manifest.json').write_text(json.dumps(dict(train=len(train),validation=len(validation),test=len(test),
            writers=names,reference_pool='Top10 train ONLY, same writer/script, strictly different characters',
            generator_writer_id=False,source=str(source),img_id_contract='reuse audited mapped IDs and existing VAE latents'),ensure_ascii=False,indent=2))
        run([sys.executable,'-u','-m','tools.reference_style.cache','--csv',str(out/'train.csv'),
             '--out',str(out/'features.npz')],'cache')
        base=json.loads((Path(a.pretrain)/'config.json').read_text())
        base.update(reference_cache=str(out/'features.npz'),data_csv=str(out/'train.csv'),
            eval_csv=str(out/'validation.csv'),output=str(out/'runs'),initialize_from=a.checkpoint,
            initialize_state='ema',max_steps=a.steps,stop_at_step=a.steps,lr_decay_steps=a.steps,
            lr=2e-5,warmup=200,ckpt_every=5000,eval_every=5000,eval_before_training=True,ema_start_step=0)
        Config(**base).validate();(out/'config.json').write_text(json.dumps(base,indent=2))
        run([sys.executable,'-u','-m','src.unet.train','--config',str(out/'config.json')],'training')
        status['phase']='completed';save()
    except BaseException as error:
        status.update(phase='failed',error=repr(error));save();raise


if __name__=='__main__':main()
