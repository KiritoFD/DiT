"""Independent U-Net construction, shared TrainState/training_loop lifecycle."""
import argparse
import copy
from dataclasses import asdict
from datetime import datetime
import json
import logging
import math
from pathlib import Path
import random
from time import perf_counter
import numpy as np
import torch
from .data import loader,StdBank
from .io import Config,ARCHITECTURE,build,eval_args,diffusion,evaluate


def parse():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--batch','--global-batch-size',dest='batch',type=int)
    p.add_argument('--max-steps',type=int);p.add_argument('--stop-at-step',type=int)
    p.add_argument('--lr-decay-steps',type=int);p.add_argument('--lr',type=float)
    p.add_argument('--warmup','--warmup-steps',dest='warmup',type=int)
    p.add_argument('--ckpt-every',type=int);p.add_argument('--eval-every',type=int)
    p.add_argument('--benchmark',type=int,default=0,help='Synthetic memory/throughput probe; no saved training')
    p.add_argument('--resume','--resume-full',dest='resume',default='');a=p.parse_args()
    config=Config(**json.loads(Path(a.config).read_text(encoding='utf-8')))
    for key in ('batch','max_steps','stop_at_step','lr_decay_steps','lr','warmup','ckpt_every','eval_every'):
        value=getattr(a,key)
        if value is not None:setattr(config,key,value)
    config.validate();return config,a


def loss_for(model,process,z,g,writer,script,objective):
    if objective!='flow':raise ValueError('U-Net entry is FM only')
    t=process.sample_t(len(z),z.device)
    with torch.autocast(z.device.type,dtype=torch.bfloat16,enabled=z.is_cuda):
        terms=process.training_losses(model,z,t,model_kwargs={'g':g,'y_callig':writer,'y_script':script})
        loss=terms['loss'].mean()
    return loss,t


def benchmark(config,steps):
    # Actual optimizer/backward/EMA; synthetic inputs isolate GPU batch capacity.
    layout=torch.channels_last if config.memory_format=='channels_last' else torch.contiguous_format
    model=build(config,initialize_tables=True).cuda().to(memory_format=layout)
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    from src.train.ema import update_ema
    opt=torch.optim.AdamW(model.parameters(),lr=config.lr,fused=True)
    process=diffusion(config);b=config.batch
    compile_start=perf_counter()
    forward=torch.compile(model,mode='default') if config.compile else model
    z=torch.randn(b,4,32,32,device='cuda').to(memory_format=layout)
    g=torch.rand(b,1,64,64,device='cuda')
    w=torch.arange(b,device='cuda')%config.num_writers;s=w%config.num_scripts
    start=None
    for i in range(steps+3):
        if i==3:torch.cuda.synchronize();start=perf_counter();torch.cuda.reset_peak_memory_stats()
        opt.zero_grad(set_to_none=True)
        loss,_=loss_for(forward,process,z,g,w,s,config.objective)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),config.grad_clip,error_if_nonfinite=True)
        opt.step()
        if i%config.ema_interval==0:update_ema(ema,model,config.ema_decay**config.ema_interval)
    torch.cuda.synchronize()
    result={'batch':b,'steps':steps,'step_s':steps/(perf_counter()-start),
        'memory_format':config.memory_format,'compile':config.compile,
        'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,
        'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30,
        'parameters':sum(p.numel() for p in model.parameters()),'final_loss':float(loss.detach()),
        'startup_and_warmup_seconds':start-compile_start}
    print('BENCHMARK '+json.dumps(result),flush=True)


def main():
    config,cli=parse()
    torch.set_num_threads(8);torch.backends.cudnn.benchmark=True
    torch.manual_seed(config.seed);torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed);random.seed(config.seed)
    if cli.benchmark:return benchmark(config,cli.benchmark)
    run=Path(config.output)/(datetime.now().strftime('%Y%m%d-%H%M%S')+'-cnn-unet-'+config.objective)
    run.mkdir(parents=True,exist_ok=False);(run/'checkpoints').mkdir()
    (run/'resolved_config.json').write_text(json.dumps(asdict(config),indent=2),encoding='utf-8')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',
        handlers=[logging.StreamHandler(),logging.FileHandler(run/'train.log',encoding='utf-8')])
    log=logging.getLogger('unet')
    from src.unet.io import training_args
    args=training_args(config,run)
    from src.train.manifest import write_source_manifest
    artifacts={k:getattr(config,k) for k in
        ('data_csv','eval_csv','vae_path','std_shape_cache','latent_shards_dir','std_soft_root',
         'callig_remap_json','font_remap_json','char_remap_json')}
    for name in ('callig','font'):
        artifacts[name+'_table']=str(Path(config.condition_tables_root)/f'{name}_table.npy')
    write_source_manifest(str(run),artifacts=artifacts)
    resume=None;start=0
    if cli.resume:
        resume=torch.load(cli.resume,map_location='cpu',weights_only=False)
        if resume.get('architecture')!=ARCHITECTURE:
            raise ValueError('Refuse a non-U-Net checkpoint')
        if resume['config']['objective']!=config.objective:
            raise ValueError('Objective changed on resume')
        start=int(resume['train_steps'])
    dataset,sampler,batches,tick,inline=loader(config,args,log,start)
    bank=StdBank(config)
    layout=torch.channels_last if config.memory_format=='channels_last' else torch.contiguous_format
    raw=build(config,initialize_tables=resume is None).cuda().to(memory_format=layout)
    ema=copy.deepcopy(raw).eval().requires_grad_(False)
    opt=torch.optim.AdamW(raw.parameters(),lr=config.lr,weight_decay=config.weight_decay,fused=True)
    from src.train.step import build_cosine_scheduler,NonFiniteGuard
    scheduler=build_cosine_scheduler(opt,warmup_steps=config.warmup,
        total_steps=config.lr_decay_steps or config.max_steps,min_lr_ratio=config.min_lr_ratio)
    if resume is not None:
        state=resume['delta']
        state={k.removeprefix('_orig_mod.'):v for k,v in state.items()}
        raw.load_state_dict(state,strict=True);ema.load_state_dict(resume['ema'],strict=True)
        opt.load_state_dict(resume['opt']);scheduler.load_state_dict(resume['scheduler'])
        from src.train.ckpt import restore_rng
        restore_rng(resume.get('rng',{}),log)
        log.warning('Shared optimizer/RNG resume; do not claim exact DataLoader replay')
    limit=config.stop_at_step or config.max_steps
    if start>=limit:raise ValueError('Resume is already at the requested stop')
    from src.train.trainer import TrainState,training_loop
    from src.train.checkpoint import CheckpointManager
    checkpoint=CheckpointManager(str(run/'checkpoints'),logger=log,strict=True)
    zero=lambda:torch.zeros((),device='cuda')
    state=TrainState(args=args,model=torch.compile(raw) if config.compile else raw,
        opt=opt,scheduler=scheduler,ema_model=ema,diffusion=diffusion(config),
        loader=batches,sampler=sampler,device=torch.device('cuda'),rank=0,logger=log,
        checkpoint_dir=str(run/'checkpoints'),_ckpt=checkpoint,
        _std_soft_cond=True,_train_vae_enc=False,_aux_dirs=[],
        _vae_sf=config.vae_scaling_factor,_vae_shift=config.vae_shift_factor,
        _epoch_steps=tick,_epochs_needed=math.ceil((limit-start)/len(batches)),
        _EVAL_INLINE=inline,_es_check_every=tick,_guard=NonFiniteGuard(log,0),
        _inj_stats={},cond_mode='2cond',train_steps=start,resume_start_step=start,
        log_steps=0,running_loss=zero(),running_diff=zero(),running_repa=zero(),
        running_skel=zero(),running_std_mid=zero(),start_time=__import__('time').time(),
        trainable_params_list=list(raw.parameters()),condition_transform=bank.transform)
    log.info('RUN=%s params=%d batch=%d max_steps=%d stop_at=%d shared=TrainState/training_loop',
        run,sum(p.numel() for p in raw.parameters()),config.batch,config.max_steps,limit)
    try:
        fatal=training_loop(state)
        if fatal is not None:raise fatal
    finally:
        checkpoint.finalize()
    log.info('DONE checkpoint_step=%s',checkpoint.last_saved)


if __name__=='__main__':main()
