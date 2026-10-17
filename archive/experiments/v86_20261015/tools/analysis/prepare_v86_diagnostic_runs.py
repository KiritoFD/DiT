"""Prepare matched 5k single-loss runs using measured gradient calibration."""
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

def main():
    audit=json.loads((ROOT/'assets/v85_diagnostics_20261014/audit.json').read_text())
    base=audit['runs']['v85E30']['config']
    # Use only keys accepted in the original configuration; resolved configs
    # also contain runtime-only fields which are not valid config arguments.
    config=json.loads((ROOT/'src/train/configs/v85E_critic.json').read_text())
    keys=['lr','weight_decay','global_seed','global_batch_size','use_c2ot','w_repa',
          'vae_scaling_factor','vae_shift_factor','warm_start_from','ema_decay',
          'ema_interval','warmup_steps','min_lr_ratio','lr_schedule']
    for k in keys:
        if k in base: config[k]=base[k]
    probe=audit['gradient_probe']
    weights={'control':(0.,0.),
             'writer':(probe['writer_weight_for_mean_25percent'],0.),
             'brush':(0.,probe['away_weight_for_mean_10percent'])}
    for arm,(w,a) in weights.items():
        c=dict(config)
        c.update(experiment_name=f'v86-diag-{arm}-5k',results_dir=f'assets/results/v86_diag_{arm}',
                 max_steps=5000,epochs=5000,ckpt_every=2500,epoch_steps=2500,probe_style_grad_steps=400,
                 compile=True,w_critic_writer=w,w_critic_away=a)
        c['lr_schedule']='cosine'
        c['warmup_steps']=1000
        c['_comment']='Matched 5k diagnostic; all arms same seed0, batch192, AdamW lr1e-5 wd0.02, warmup1000 then cosine to0.1, fresh optimizer from v84A200k. Historical initial400-step mean gradient calibration, not guaranteed per-step target; verify probes before interpreting results.'
        p=ROOT/'src/train/configs'/f'v86_diag_{arm}_5k.json'
        p.write_text(json.dumps(c,ensure_ascii=False,indent=2),encoding='utf-8')
        from src.train.cli import parse_args
        args=parse_args(['--config',str(p)])
        assert args.max_steps==5000 and args.global_batch_size==192
        assert args.epoch_steps==args.ckpt_every==2500
        assert args.global_seed==0 and args.w_critic_writer==w and args.w_critic_away==a
        print(p,w,a)

if __name__=='__main__':main()
