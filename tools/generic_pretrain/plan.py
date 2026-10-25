"""Emit explicit budgets and matching generic/Writer-only/scratch configs."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from src.unet.io import Config
from tools.cohort_eval.datasets import sha


def create(args):
    data=Path(args.data);manifest=json.loads((data/'manifest.json').read_text())
    if manifest['status']!='complete':raise ValueError('Data preparation incomplete')
    for name,digest in manifest['files'].items():
        if sha(data/name)!=digest:raise ValueError('Prepared data changed: '+name)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);tables=data/'tables'
    common=dict(channels=args.channels,num_res_blocks=args.blocks,num_writers=20,num_scripts=3,
        batch=args.batch,seed=90,compile=True,freeze_writer_table=True,writer_table_file=args.writer_table,
        decouple_writer_condition=True,std_shape_aug='off',std_soft_sigma_min=0.,std_soft_sigma_max=0.,std_soft_eval_sigma=0.,
        std_soft_root=manifest['std_root'],vae_path=args.vae,vae_scaling_factor=.18215,vae_shift_factor=0.,
        latent_shards_dir=str(data/'latents'),img_root='',condition_tables_root=str(tables),
        **{f'{name}_remap_json':str(tables/f'{name}_remap.json') for name in ('callig','font','char')})
    pre=Config(**common,writer_mode='none',lr=7e-5,warmup=min(1500,args.pretrain_steps-1),
        max_steps=args.pretrain_steps,lr_decay_steps=args.pretrain_steps,
        data_csv=str(data/'generic_train.csv'),eval_csv=str(data/'generic_validation.csv'),output=str(out/'runs/pretrain'))
    fine=Config(**common,writer_mode='film',writer_film_rank=16,writer_film_layers=('middle','decoder'),
        lr=5e-5,warmup=min(200,args.finetune_steps-1),max_steps=args.finetune_steps,lr_decay_steps=args.finetune_steps,
        adapter_only_steps=args.adapter_steps,backbone_lr_scale=.2,generic_replay_fraction=args.replay,
        finetune_backbone_patterns=args.joint_patterns,
        data_csv=str(data/('finetune_mixed.csv' if args.replay else 'top20_train.csv')),
        eval_csv=str(data/'top20_validation.csv'),output=str(out/'runs/finetune'))
    # Same model/data path without pretraining; scratch must learn the entire initially random backbone.
    scratch=Config(**{**asdict(fine),'adapter_only_steps':0,'finetune_backbone_patterns':['*'],
        'generic_replay_fraction':0.,'data_csv':str(data/'top20_train.csv'),'output':str(out/'runs/scratch')})
    for name,config in (('pretrain',pre),('finetune',fine),('scratch',scratch)):
        config.validate();(out/f'{name}.json').write_text(json.dumps(asdict(config),indent=2))
    plan=dict(data=str(data),data_manifest_sha256=sha(data/'manifest.json'),config_dir=str(out),
        pretrain_steps=args.pretrain_steps,finetune_steps=args.finetune_steps,
        control_caveat='scratch trains all layers; default finetune has adapter warmup/partial unfreeze. '
        'To isolate pretraining alone, set adapter_steps=0 and full unfreeze for BOTH arms.',
        encode=[str(data/'encode.csv'),str(data/'latents'),args.vae],
        status='prepared_not_started')
    (out/'plan.json').write_text(json.dumps(plan,indent=2));print(out/'plan.json')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('data','out','writer-table'):p.add_argument('--'+key,required=True)
    p.add_argument('--pretrain-steps',type=int,required=True);p.add_argument('--finetune-steps',type=int,required=True)
    p.add_argument('--adapter-steps',type=int,default=1000);p.add_argument('--replay',type=float,default=.1)
    p.add_argument('--joint-patterns',nargs='+',default=['decoder.','middle','up.','output.'])
    p.add_argument('--channels',nargs=3,type=int,default=[192,384,576]);p.add_argument('--blocks',type=int,default=4)
    p.add_argument('--batch',type=int,default=512);p.add_argument('--vae',default='pretrained_models/sd-vae-ft-ema')
    args=p.parse_args()
    if min(args.pretrain_steps,args.finetune_steps)<2 or not 0<=args.adapter_steps<args.finetune_steps:p.error('Invalid stage budgets')
    create(args)


if __name__=='__main__':main()
