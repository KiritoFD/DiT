"""Independent U-Net configuration/checkpoint identity and shared evaluation adapter."""
import argparse
from dataclasses import dataclass,asdict
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from .model import LatentUNet

ARCHITECTURE='latent_cnn_unet_v1'


@dataclass
class Config:
    architecture:str=ARCHITECTURE
    objective:str='flow'
    channels:tuple=(128,256,384)
    num_res_blocks:int=2
    decoder_res_blocks:tuple|None=None  # 32/16/8 order; None inherits encoder depth.
    skip_gate_scales:tuple=()
    std_gate_scales:tuple=()
    requires_warm_start:bool=False
    num_writers:int=10
    num_scripts:int=3
    batch:int=128
    reference_batch:int=192
    memory_format:str='channels_last'
    condition_tables_root:str='assets/triple_tables_best_minimal'
    max_steps:int=0  # Required from config/CLI; no implicit scientific budget.
    stop_at_step:int=0
    lr_decay_steps:int=0  # 0 means max_steps, never stop_at_step.
    checkpoint_steps:tuple=()
    workers:int=8
    seed:int=90
    lr:float=5e-5
    weight_decay:float=.02
    warmup:int=1500
    min_lr_ratio:float=.1
    grad_clip:float=1.
    ema_decay:float=.999
    ema_interval:int=2
    ema_warmup:bool=True
    ema_start_step:int=0
    log_every:int=50
    eval_every:int=5000
    ckpt_every:int=5000
    output:str='assets/results/v91_unet_fm_std'
    data_csv:str='exp-std/csv/train_top10_aug_sym.csv'
    latent_shards_dir:str='exp-std/data/shards_img_aug'
    img_root:str='data/top10_style23/imgs'
    std_soft_root:str='exp-std/data/std_soft_glyphs_v1'
    std_shape_cache:str='assets/std_shape_bank_64.pt'
    std_shape_aug:str='strong'
    char_remap_json:str='assets/triple_tables_best_minimal/char_remap.json'
    callig_remap_json:str='assets/triple_tables_best_minimal/callig_remap.json'
    font_remap_json:str='assets/triple_tables_best_minimal/font_remap.json'
    eval_csv:str='exp-std/csv/eval200_fixed.csv'
    vae_path:str='pretrained_models/sd-vae-ft-ema'
    vae_scaling_factor:float=.18215
    vae_shift_factor:float=0.
    eval_batch:int=16
    eval_steps:int=50
    compile:bool=False

    def validate(self,require_budget=True):
        if self.architecture!=ARCHITECTURE:raise ValueError('Not a CNN U-Net configuration')
        if self.memory_format not in ('channels_last','nchw'):raise ValueError(self.memory_format)
        if self.objective!='flow':raise ValueError('First U-Net training line supports FM only')
        if self.std_shape_aug not in ('off','mild','strong'):raise ValueError(self.std_shape_aug)
        sizes=(self.batch,self.log_every,self.ema_interval,self.ckpt_every,self.eval_every)
        if min(sizes+((self.max_steps,) if require_budget else ()))<=0:
            raise ValueError('Batch, budget and intervals must be positive')
        if self.eval_every!=self.ckpt_every:raise ValueError('Shared trainer uses one checkpoint/eval interval')
        if self.stop_at_step and not 0<self.stop_at_step<=self.max_steps:raise ValueError('Invalid phase stop')


def build(config,initialize_tables=False):
    config.validate(require_budget=False)
    model=LatentUNet(tuple(config.channels),config.num_writers,config.num_scripts,
                    num_res_blocks=getattr(config,'num_res_blocks',2),
                    decoder_res_blocks=config.decoder_res_blocks,
                    skip_gate_scales=config.skip_gate_scales,std_gate_scales=config.std_gate_scales)
    if initialize_tables:
        import numpy as np
        for name,embedding in [('callig',model.writer),('font',model.script)]:
            values=torch.from_numpy(np.load(Path(config.condition_tables_root)/f'{name}_table.npy'))
            if values.shape!=embedding.weight[:-1].shape:raise ValueError(f'{name} table shape mismatch')
            with torch.no_grad():embedding.weight[:-1].copy_(values)
    return model


def eval_args(config):
    return SimpleNamespace(**asdict(config),model=ARCHITECTURE,global_seed=config.seed,
        std_soft_cond=True,use_script_cond=True,latent_channels=4,image_size=256,
        diffusion_type=config.objective,eval_cfg=1.,flow_sampler='heun',sde_gamma=0.,shift=1.,
        in_mem_eval_sets=f'eval200fix:{config.eval_csv}:187',in_mem_eval_save=True,
        in_mem_eval_batch=config.eval_batch,in_mem_eval_vae_batch=4,std_soft_eval_sigma=.65,
        results_dir=config.output,eval_dir=config.output,w_repa=0.,
        eval_vae_path=config.vae_path,
        t_sampler='logit_normal',t_mean=0.,t_std=1.)


def training_args(config,run):
    """Adapter to the same argument names consumed by the existing trainer."""
    args=eval_args(config)
    args.architecture_config=asdict(config)
    args.train_path='latent-std'
    args.global_batch_size=config.batch
    args.num_workers=config.workers
    args.preload=True;args.preload_workers=8
    args.repa_need_image='never'
    args.sampler='uniform';args.epoch_steps=config.ckpt_every
    args.eval_mode='inline';args.in_mem_eval=True
    args.eval_dir=str(run);args.results_dir=str(run)
    return args


def diffusion(config,evaluation=False):
    from src.loss.flow_matching import FlowMatching
    if config.objective!='flow':raise ValueError('First U-Net line is FM only')
    return FlowMatching(num_steps=config.eval_steps,sampler='heun')


def checkpoint_config(ck):
    """Legacy U-Net identity lives in args; never guess an unknown architecture."""
    saved=ck.get('args')
    get=lambda key: saved.get(key) if isinstance(saved,dict) else getattr(saved,key,None)
    architecture=ck.get('architecture') or get('architecture')
    if architecture!=ARCHITECTURE:raise ValueError('Refuse a DiT/unknown checkpoint')
    values=ck.get('config') or get('architecture_config')
    config=Config(**values)
    config.validate(require_budget=False)
    return config


def load(path,device='cpu',use_ema=True):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    config=checkpoint_config(ck)
    if use_ema and 'architecture' not in ck:
        raise ValueError('Legacy EMA is unverified; explicitly load raw instead')
    model=build(config).to(device)
    key='ema' if use_ema else 'delta'
    weights={name.removeprefix('_orig_mod.'):value for name,value in ck[key].items()}
    model.load_state_dict(weights,strict=True);model.eval()
    return model,config,ck


def load_model_from_ckpt(path,device='cpu',use_ema=True):
    """Same public loading interface as the DiT evaluator, separate strict builder."""
    model,config,_=load(path,device,use_ema)
    args=eval_args(config)
    args.eval_use_ema=use_ema
    return model,args


def evaluate(model,config,step,directory,*,use_ema=True):
    from src.eval.in_mem_eval import run_in_mem_eval
    args=eval_args(config)
    args.eval_use_ema=use_ema
    return run_in_mem_eval(model,args,step,next(model.parameters()).device,
        results_dir=str(directory),diffusion_override=diffusion(config,evaluation=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True)
    p.add_argument('--out',required=True);a=p.parse_args()
    model,config,ck=load(a.checkpoint,'cuda')
    with torch.no_grad():evaluate(model,config,ck['train_steps'],a.out)
