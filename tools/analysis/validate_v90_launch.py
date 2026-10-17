"""Validate phase identity and real cache before GPU allocation; emit native full-batch smoke."""
import argparse
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.train.cli import parse_args


def validate(args,checkpoint=None):
    if args.model!='DiT-2Cond-S384-20/2' or args.max_steps!=200000:
        raise ValueError('20L model and200k global LR horizon required')
    if args.warm_start_from or args.pretrained:raise ValueError('No12L initialization')
    if args.std_shape_aug not in ('strong','mild'):raise ValueError('Explicit shape augmentation mode required')
    if not Path(args.std_shape_cache).is_file():raise FileNotFoundError(args.std_shape_cache)
    if args.std_soft_bias or not args.std_soft_zero_init or args.glyph_scale_bound!=.5:
        raise ValueError('Missing STD stability contract')
    for key in ('pair_fraction','rank_gradient_ratio','crosschar_gradient_ratio','w_style_rank',
                'w_critic_writer','w_critic_away','deform_skel','w_latent_skel','w_std_mid'):
        if getattr(args,key,0):raise ValueError('Unrequested auxiliary intervention: '+key)
    if getattr(args,'strong_style_mode','off')!='off':raise ValueError('Strong-style supervision must be disabled')
    if checkpoint is not None:
        old=checkpoint['args'];old=vars(old) if not isinstance(old,dict) else old
        previous={50000:5000,200000:50000}.get(args.stop_at_step)
        if checkpoint.get('train_steps',checkpoint.get('step'))!=previous:raise ValueError('Unexpected prior phase step')
        for key in ('model','std_shape_aug','global_batch_size','max_steps','lr','warmup_steps'):
            if old[key]!=getattr(args,key):raise ValueError('Resume changed experimental identity: '+key)
        if checkpoint.get('opt') is None or checkpoint.get('scheduler') is None:
            raise ValueError('Full optimizer/scheduler state required, not an EMA-only weight file')
    elif args.stop_at_step!=5000:raise ValueError('Later phases require explicit resume-full checkpoint')
    if args.stop_at_step%args.ckpt_every:raise ValueError('Phase boundary must save a complete checkpoint')


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--resume',default='')
    p.add_argument('--smoke-out',default='');a=p.parse_args();args=parse_args(['--config',a.config])
    checkpoint=torch.load(a.resume,map_location='cpu',weights_only=False) if a.resume else None
    validate(args,checkpoint)
    if a.smoke_out:
        c=json.loads(Path(a.config).read_text(encoding='utf-8'))
        c.update(stop_at_step=4,ckpt_every=4,epoch_steps=4,log_every=1,std_shape_log_every=1,
            experiment_name='v90-20L-technical-smoke',results_dir='assets/results/v90_20l_technical_smoke',
            in_mem_eval_sets='probe:exp-std/csv/eval200_fixed.csv:11')
        out=Path(a.smoke_out)
        if out.exists():raise FileExistsError(out)
        out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(c,indent=2),encoding='utf-8')
    print('V90_PHASE_VALIDATED',flush=True)


if __name__=='__main__':main()
