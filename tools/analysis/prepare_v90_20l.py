"""Materialize reviewed20L strong/mild phase configs; never submit GPU work."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def configuration(mode='strong',stop=5000,batch=192):
    if mode not in ('strong','mild'):raise ValueError(mode)
    c=json.loads((ROOT/'src/train/configs/v84A_std_cond.json').read_text(encoding='utf-8'))
    label='v90A_20L_stdstrong' if mode=='strong' else 'v90B_20L_stdmild'
    c.update(_comment='Fresh20L backbone, width384/heads8/patch2. Batch provisional; GPU calibration required. '
        'No P1 pair/rank, Critic, or predicted skeleton. Phase stops preserve200k LR horizon. '
        'Mild control shares clean probabilities but maps every non-clean tier to mild.',
        experiment_name=label,results_dir='assets/results/'+label,model='DiT-2Cond-S384-20/2',
        pretrained=None,warm_start_from='',resume_full=None,reset_cond_head=False,
        cond_inject_at='3,6,9,12,15',std_soft_bias=False,glyph_scale_init=.2,glyph_scale_bound=.5,
        glyph_vec_cond=True,glyph_vec_dim=128,glyph_vec_pool='mean',glyph_drop_prob=.05,
        std_shape_aug=mode,std_shape_cache='assets/std_shape_bank_64.pt',std_shape_log_every=500,
        std_soft_sigma_min=.65,std_soft_sigma_max=.65,std_soft_eval_sigma=.65,
        max_steps=200000,stop_at_step=stop,epochs=200000,global_batch_size=batch,
        ckpt_every=1000,epoch_steps=1000,ckpt_keep=0,in_mem_eval_batch=11,in_mem_eval_vae_batch=8,
        in_mem_eval_poster=False,log_every=50,rank_gradient_ratio=0.,pair_fraction=0.,
        strong_style_mode='off',w_critic_writer=0.,w_critic_away=0.,w_style_rank=0.,
        glyph_deform_prob=0.,glyph_noise_prob=0.,glyph_patch_drop=0.,glyph_mask_prob=0.)
    return c


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='src/train/configs/v90_20l')
    p.add_argument('--batch',type=int,default=192);p.add_argument('--force',action='store_true')
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    for mode in ('strong','mild'):
        for stop in (5000,50000,200000):
            path=out/f'{mode}_{stop//1000}k.json'
            if path.exists() and not a.force:raise FileExistsError(f'Refusing to overwrite reviewed config: {path}')
            path.write_text(json.dumps(configuration(mode,stop,a.batch),ensure_ascii=False,indent=2),encoding='utf-8')
    print('V90_20L_CONFIGS_READY',out,'GPU_PREFLIGHT_NOT_RUN',flush=True)


if __name__=='__main__':main()
