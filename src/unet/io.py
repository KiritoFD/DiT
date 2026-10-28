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
    writer_film_rank:int=0  # 0=legacy shared FiLM only; >0 adds Writer-only residual FiLM.
    writer_film_layers:tuple=()  # Layer name patterns, e.g. ('decoder.1', 'decoder.2'); empty means all.
    writer_film_spec:dict|None=None  # Explicit mapping: {pattern: max_amp}, e.g. {'middle': 0.3, 'decoder.2': 0.3, ...}
    decouple_writer_condition:bool=False  # Base condition is purely MLP(t + script); Writer is exclusively in FiLM.
    decoder_std_residual:bool=False  # Direct 16² & 32² lightweight STD residual into Decoder.
    decoder_std_scale:float=0.2  # Scaling factor for Decoder STD residual.
    residual_cond_gate:bool=False  # Per-channel residual gains conditioned on shared t/w/s.
    stage_specific_cond:bool=False  # DiC-style multi-scale condition injection (32/16/8).
    stage_cond_mode:str='adapter'  # 'adapter' (zero-init delta) or 'independent'
    act_fn:str='silu'  # 'silu' or 'gelu'
    requires_warm_start:bool=False
    num_writers:int=10
    freeze_writer_table:bool=True
    writer_mode:str='shared'  # none=generic; film=Writer-only, hard-zero for CFG-null/replay
    reference_dim:int=0
    reference_cache:str=''
    reference_max_refs:int=2
    reference_con_weight:float=.05
    reference_temperature:float=.1
    reference_hard_weight:float=2.
    reference_injection:str='add'  # 'add' legacy single-vector; 'writer_slot' rides the FiLM reader pathway.
    reference_dropout:float=0.  # Probability of zeroing the aggregate style (trains the null-style path).
    reference_runtime:str='default'  # 'default' legacy style_writer_* labels; 'group' pipeline group ids.
    reference_extra_cache:str=''  # optional second bank (e.g. Top10) merged for group runtime refs.
    brush_dim:int=0
    brush_dropout:float=.1
    brush_noise:float=.075
    brush_cache:str=''
    brush_generic_fraction:float=.75
    drop_writer:float=.08
    writer_cond_lr:float=0.  # >0: writer embedding + condition projections use this LR
    pair_fraction:float=0.  # >0: PairMixtureSampler same-char cross-writer pairs share of batch
    null_writer_rows:bool=False  # allow training rows explicitly mapped to the null writer row
    semantic_writer_rows:dict|None=None  # raw semantic Writer ID -> checkpoint embedding row; never infer from 0..9.
    study_spec:str=''  # opt-in experiment adapter; legacy/mainline behavior unchanged when empty.
    initialize_from:str=''  # model-only stage transfer; new optimizer/scheduler/step clock
    initialize_state:str='ema'
    adapter_only_steps:int=0
    finetune_backbone_patterns:tuple=('decoder.','middle','up.','output.')
    backbone_lr_scale:float=.2
    finetune_condition_lr_scale:float|None=None  # Optional separate STD/Script group relative to adapter LR.
    generic_replay_fraction:float=0.
    writer_table_file:str=''
    eval_before_training:bool=False
    drop_path_max:float=0.
    drop_path_start:int=2000
    drop_path_end:int=5000
    grouped_weight_decay:bool=False
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
    std_soft_sigma_min:float=.4
    std_soft_sigma_max:float=1.
    std_soft_eval_sigma:float=.65
    std_shape_cache:str='assets/std_shape_bank_64.pt'
    std_shape_aug:str='strong'
    std_shape_step_offset:int=0  # Preserve the source-stage augmentation schedule on model-only transfer.
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
    style_matrix_repr:int=-1
    style_matrix_injection:int=0
    style_matrix_table:str=''
    style_matrix_vector_dim:int=32
    microbatch:int=0  # 0 = global batch; matrix runtime accumulates optimizer steps.
    writer_velocity:str=''  # cnn=S9, geometry=S10, attention=S11

    def validate(self,require_budget=True):
        if self.style_matrix_repr>=0:
            if self.style_matrix_repr not in range(5) or self.style_matrix_injection not in range(4):
                raise ValueError('Matrix requires R0..R4 and I0..I3')
            if self.reference_dim or self.brush_dim or self.writer_film_rank or self.stage_specific_cond:
                raise ValueError('Matrix requires a plain common content backbone')
            if self.writer_mode!='shared' or self.decouple_writer_condition or self.decoder_std_residual or self.skip_gate_scales or self.std_gate_scales:
                raise ValueError('Matrix content route must preserve the plain P1 topology')
            if self.microbatch and (self.microbatch>self.batch or self.microbatch<1):
                raise ValueError('Matrix microbatch must be in [1, global batch]')
        if self.reference_dim:
            if self.reference_dim!=128 or self.brush_dim:
                raise ValueError('Reference route requires 128D style, no Brush condition')
            if self.writer_mode=='shared' and self.reference_injection!='writer_slot':
                raise ValueError('Reference on a shared-writer model requires writer_slot injection')
            if self.writer_mode not in ('none','shared'):
                raise ValueError('Reference route requires writer_mode none or shared')
            if not self.reference_cache or not 1<=self.reference_max_refs<=4 or self.batch%4:
                raise ValueError('Reference route needs feature bank, K=1..4, batch multiple of 4')
            if self.reference_con_weight<=0 or self.reference_temperature<=0 or self.reference_hard_weight<1:
                raise ValueError('Invalid reference contrastive settings')
            if self.reference_injection not in ('add','writer_slot'):
                raise ValueError('Invalid reference_injection')
            if not 0<=self.reference_dropout<1:raise ValueError('Invalid reference dropout')
            if self.generic_replay_fraction:
                raise ValueError('Reference route does not use legacy Writer-ID replay')
        if self.writer_mode not in ('shared','none','film'):raise ValueError('Invalid writer_mode')
        if self.brush_dim not in (0,5) or not 0<=self.brush_dropout<1 or self.brush_noise<0:
            raise ValueError('Invalid Brush condition')
        if not 0<=self.drop_writer<1:raise ValueError('Invalid writer drop probability')
        if not 0<=self.pair_fraction<0.5:raise ValueError('Invalid pair fraction')
        if self.writer_cond_lr<0:raise ValueError('Invalid writer condition LR')
        if self.semantic_writer_rows is not None:
            rows=[int(v) for v in self.semantic_writer_rows.values()]
            if len(set(rows))!=len(rows) or any(v<0 or v>=self.num_writers for v in rows):
                raise ValueError('Invalid checkpoint semantic Writer mapping')
        if self.null_writer_rows and self.writer_mode!='shared':raise ValueError('null_writer_rows needs shared route')
        slot=self.reference_dim and self.reference_injection=='writer_slot'
        if self.writer_mode!='shared':
            if self.architecture!=ARCHITECTURE:raise ValueError('Plain U-Net architecture required')
            if slot:
                if self.decouple_writer_condition:
                    raise ValueError('writer_slot injection needs coupled conditions so writer_proj stays live')
            elif not self.decouple_writer_condition:
                raise ValueError('Generic/Writer-only route requires decoupled plain U-Net')
            if self.writer_mode=='none' and self.writer_film_rank:raise ValueError('Generic stage may not contain Writer FiLM')
            if self.writer_mode=='film' and not self.writer_film_rank:raise ValueError('Writer-only stage needs Writer FiLM')
            if not self.freeze_writer_table:raise ValueError('Writer tables remain independently pretrained and frozen')
            if self.stage_specific_cond:raise ValueError('Two-stage baseline does not change stage-specific conditioning')
        if self.adapter_only_steps<0 or not 0<self.backbone_lr_scale<=1:raise ValueError('Invalid adaptation schedule')
        if self.finetune_condition_lr_scale is not None and not 0<self.finetune_condition_lr_scale<=1:
            raise ValueError('Invalid STD/Script learning-rate scale')
        if not 0<=self.generic_replay_fraction<1:raise ValueError('Invalid generic replay fraction')
        if self.initialize_state not in ('raw','ema'):raise ValueError('Invalid initialization state')
        if (self.adapter_only_steps or self.generic_replay_fraction) and self.writer_mode!='film':
            raise ValueError('Adaptation/replay require Writer-only route')
        if not 0<=self.std_soft_sigma_min<=self.std_soft_sigma_max or self.std_soft_eval_sigma<0:
            raise ValueError('Invalid STD blur range')
        if not 0<=self.drop_path_max<1 or not 0<=self.drop_path_start<self.drop_path_end:
            raise ValueError('Invalid DropPath configuration')
        if not isinstance(self.freeze_writer_table,bool):raise ValueError('freeze_writer_table must be boolean')
        if self.writer_table_file and (not self.freeze_writer_table or self.architecture!=ARCHITECTURE):
            raise ValueError('External writer table requires frozen latent_cnn_unet_v1 writer rows')
        if self.architecture not in (ARCHITECTURE, 'calliflow_v2'):
            raise ValueError(f'Unsupported architecture: {self.architecture}')
        if not isinstance(self.writer_film_rank,int) or self.writer_film_rank<0:
            raise ValueError('writer_film_rank must be a nonnegative integer')
        if self.writer_film_layers is not None and not isinstance(self.writer_film_layers,(tuple,list)):
            raise ValueError('writer_film_layers must be a tuple/list of strings or empty')
        if self.writer_film_spec is not None and not isinstance(self.writer_film_spec,dict):
            raise ValueError('writer_film_spec must be a dict {pattern: max_amp} or None')
        if not isinstance(self.decouple_writer_condition,bool):
            raise ValueError('decouple_writer_condition must be boolean')
        if not isinstance(self.decoder_std_residual,bool):
            raise ValueError('decoder_std_residual must be boolean')
        if not isinstance(self.residual_cond_gate,bool):raise ValueError('residual_cond_gate must be boolean')
        if not isinstance(self.stage_specific_cond,bool):raise ValueError('stage_specific_cond must be boolean')
        if self.stage_cond_mode not in ('adapter','independent'):raise ValueError(f'Invalid stage_cond_mode: {self.stage_cond_mode}')
        if self.act_fn not in ('silu','gelu'):raise ValueError(f'Invalid act_fn: {self.act_fn}')
        if self.memory_format not in ('channels_last','nchw'):raise ValueError(self.memory_format)
        if self.objective!='flow':raise ValueError('First U-Net training line supports FM only')
        if self.std_shape_aug not in ('off','mild','strong'):raise ValueError(self.std_shape_aug)
        if self.std_shape_step_offset<0:raise ValueError('Negative STD augmentation clock')
        sizes=(self.batch,self.log_every,self.ema_interval,self.ckpt_every,self.eval_every)
        if min(sizes+((self.max_steps,) if require_budget else ()))<=0:
            raise ValueError('Batch, budget and intervals must be positive')
        if self.eval_every!=self.ckpt_every:raise ValueError('Shared trainer uses one checkpoint/eval interval')
        if self.stop_at_step and not 0<self.stop_at_step<=self.max_steps:raise ValueError('Invalid phase stop')


def build(config,initialize_tables=False):
    config.validate(require_budget=False)
    if config.architecture == 'calliflow_v2':
        from .calliflow_v2 import CalliFlowV2
        model = CalliFlowV2(
            channels=tuple(config.channels),
            writer_channels=(64, 96, 128),
            num_res_blocks=getattr(config, 'num_res_blocks', 4),
            middle_res_blocks=3,
            num_writers=config.num_writers,
            num_scripts=config.num_scripts,
            cond_dim=512,
            act_fn=getattr(config, 'act_fn', 'gelu'),
        )
        if initialize_tables:
            import numpy as np
            for name, embedding in [('callig', model.writer_branch.writer), ('font', model.script)]:
                p = Path(config.condition_tables_root) / f'{name}_table.npy'
                if p.exists():
                    values = torch.from_numpy(np.load(p))
                    if values.shape == embedding.weight[:-1].shape:
                        with torch.no_grad():
                            embedding.weight[:-1].copy_(values)
        return model
    model=LatentUNet(tuple(config.channels),config.num_writers,config.num_scripts,
                    num_res_blocks=getattr(config,'num_res_blocks',2),
                    decoder_res_blocks=config.decoder_res_blocks,
                    skip_gate_scales=config.skip_gate_scales,std_gate_scales=config.std_gate_scales,
                    writer_film_rank=config.writer_film_rank,
                    writer_film_layers=getattr(config,'writer_film_layers',()),
                    writer_film_spec=getattr(config,'writer_film_spec',None),
                    decouple_writer_condition=getattr(config,'decouple_writer_condition',False),
                    decoder_std_residual=getattr(config,'decoder_std_residual',False),
                    decoder_std_scale=getattr(config,'decoder_std_scale',0.2),
                    residual_cond_gate=config.residual_cond_gate,
                    stage_specific_cond=getattr(config,'stage_specific_cond',False),
                    stage_cond_mode=getattr(config,'stage_cond_mode','adapter'),
                    act_fn=getattr(config,'act_fn','silu'),drop_path_max=config.drop_path_max,writer_mode=config.writer_mode,
                    brush_dim=config.brush_dim,brush_dropout=config.brush_dropout,brush_noise=config.brush_noise,
                    reference_dim=config.reference_dim,drop_writer=getattr(config,'drop_writer',.08),
                    reference_injection=getattr(config,'reference_injection','add'),
                    reference_dropout=getattr(config,'reference_dropout',0.))
    if initialize_tables:
        import numpy as np
        for name,embedding in [('callig',model.writer),('font',model.script)]:
            values=torch.from_numpy(np.load(Path(config.condition_tables_root)/f'{name}_table.npy'))
            if values.shape!=embedding.weight[:-1].shape:raise ValueError(f'{name} table shape mismatch')
            with torch.no_grad():embedding.weight[:-1].copy_(values)
    model.writer.weight.requires_grad_(not config.freeze_writer_table)
    if config.writer_mode!='shared' and not config.reference_dim:
        for name,p in model.named_parameters():
            if name.startswith('writer_proj.'):p.requires_grad_(False)
    if config.style_matrix_repr>=0:
        from .style.interface import MatrixUNet
        model=MatrixUNet(model,config.style_matrix_repr,config.style_matrix_injection,
                         vector_dim=config.style_matrix_vector_dim)
    if config.writer_velocity:
        if config.style_matrix_repr>=0:raise ValueError('Velocity branch cannot wrap the old matrix')
        from .velocity import WriterVelocityUNet
        model=WriterVelocityUNet(model,config.writer_velocity)
    return model


def eval_args(config):
    n_eval = 187
    if getattr(config, 'eval_csv', None) and Path(config.eval_csv).exists():
        try:
            import csv
            with open(config.eval_csv, encoding='utf-8') as f:
                n_eval = len(list(csv.DictReader(f)))
        except Exception:
            pass
    return SimpleNamespace(**asdict(config),model=ARCHITECTURE,global_seed=config.seed,
        std_soft_cond=True,use_script_cond=True,latent_channels=4,image_size=256,
        diffusion_type=config.objective,eval_cfg=1.,flow_sampler='heun',sde_gamma=0.,shift=1.,
        in_mem_eval_sets=f'eval200fix:{config.eval_csv}:{n_eval}',in_mem_eval_save=True,
        in_mem_eval_batch=config.eval_batch,in_mem_eval_vae_batch=4,
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
    args.pair_fraction=getattr(config,'pair_fraction',0.)
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
    if architecture not in (ARCHITECTURE, 'calliflow_v2'):
        raise ValueError(f'Refuse a DiT/unknown checkpoint: {architecture}')
    values=ck.get('config') or get('architecture_config')
    values=dict(values)
    values.setdefault('freeze_writer_table',False)  # historical U-Net runs trained their embeddings
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
    if config.reference_dim:
        if config.reference_runtime=='group':
            from .reference.group_runtime import MergedBank,bind_group_evaluation
            bind_group_evaluation(model,config,MergedBank(config.reference_cache,config.reference_extra_cache) if config.reference_extra_cache else MergedBank(config.reference_cache))
        else:
            from .reference.runtime import bind_evaluation
            bind_evaluation(model,config)
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
        results_dir=str(directory))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True)
    p.add_argument('--out',required=True);a=p.parse_args()
    model,config,ck=load(a.checkpoint,'cuda')
    with torch.no_grad():evaluate(model,config,ck['train_steps'],a.out)
