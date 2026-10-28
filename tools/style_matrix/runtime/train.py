"""Plain FM matrix runner, reusing repository data, STD bank, FM, EMA and ckpt."""
import argparse
import copy
import itertools
import json
import logging
from pathlib import Path
import signal
import time
import torch
from torch.utils.data import DataLoader,Sampler
from src.unet.io import Config,training_args,diffusion
from src.unet.data import loader,StdBank
from src.train.step import build_cosine_scheduler,build_restart_scheduler
from src.train.ema import update_ema
from src.train.ckpt import build_checkpoint,WRITER,restore_rng,capture_rng,drain_ckpt
from src.unet.style.diagnostics import inspect,gradient_report
from .common import construct,optimizer,synthetic,step,save,seed,sha,condition_report


class OffsetSampler(Sampler):
    def __init__(self,base,epoch,skip):self.base=base;self.epoch=epoch;self.skip=skip
    def __iter__(self):
        self.base.set_epoch(self.epoch)
        return itertools.islice(iter(self.base),self.skip,None)
    def __len__(self):return len(self.base)-self.skip


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--stop',required=True,type=int)
    p.add_argument('--resume',default='');p.add_argument('--run',required=True)
    a=p.parse_args();spec=json.loads(Path(a.spec).read_text());cfg=Config(**spec['config'])
    if getattr(cfg,'study_spec',''):raise ValueError('S/J runtime does not mix legacy study adapters')
    run=Path(a.run);run.mkdir(parents=True,exist_ok=True)
    if not 0<a.stop<=cfg.lr_decay_steps:raise ValueError('Stop outside fixed LR horizon')
    if (run/'identity.json').exists() and json.loads((run/'identity.json').read_text()) not in (
            spec['identity'],spec.get('continues_identity')):
        raise ValueError('Run directory belongs to another experiment')
    if not a.resume and any((run/'checkpoints').glob('*.pt')):
        raise FileExistsError('Existing checkpoint: resume explicitly, never restart over it')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s');log=logging.getLogger('matrix')
    torch.set_num_threads(8);torch.backends.cudnn.benchmark=True
    raw,cfg,transfer=construct(spec,initialize_source=not bool(a.resume))
    resume=torch.load(a.resume,map_location='cpu',weights_only=False) if a.resume else None
    start=int(resume['train_steps']) if resume else 0
    if resume:
        assert resume['matrix_identity']==spec.get('continues_identity',spec['identity']), 'Scientific run identity changed'
        assert start<a.stop, 'Resume already reached the requested boundary'
        raw.load_state_dict(resume['delta'],strict=True)
    else:
        with torch.autocast('cuda',dtype=torch.bfloat16):initial=condition_report(raw,cfg,calibrate=True)
        save(run/'initialization.json',dict(transfer=transfer,diagnostics=initial))
    auxiliary=None;reader=None
    if spec.get('kind')=='joint':
        import csv
        from src.unet.joint.objective import JointDecoderObjective,GTImages
        jc=spec['joint'];cal=json.loads(Path(jc['calibration']).read_text())
        assert cal['source_checkpoint_sha256']==sha(spec['source'])
        assert cal['train_csv_sha256']==sha(cfg.data_csv), 'Calibration used a different training set'
        auxiliary=JointDecoderObjective(jc['arm'],cfg,image_micro=jc['image_micro'],image_batch=jc['image_batch'],
            img_weight=cal['image_weight'],rec_weight=cal['reconstruction_weight'])
        assert cal['vae_frozen_sha256']==auxiliary.frozen_sha
        reader=GTImages(list(csv.DictReader(open(cfg.data_csv,encoding='utf-8-sig'))))
        if resume:auxiliary.restore(resume)
    ema=copy.deepcopy(raw).eval().requires_grad_(False)
    opt=optimizer(raw,cfg,auxiliary.vae.decoder if auxiliary else None,scratch=spec.get('initialization')=='scratch')
    restart=spec.get('schedule_restart')
    if restart and not resume:raise ValueError('Controlled LR restart requires --resume')
    if resume:
        ema.load_state_dict(resume['ema'],strict=True);opt.load_state_dict(resume['opt'])
    if restart:
        scheduler=build_restart_scheduler(opt,start_step=start,horizon=restart['horizon'],
            warmup_steps=restart['warmup'],peak_scale=restart['peak_scale'],min_lr_ratio=cfg.min_lr_ratio)
        scheduler.last_epoch=start;assert scheduler.last_epoch==start
        # The first restarted step keeps the pre-restart LR for exact continuity,
        # then the fresh warmup takes over on the next scheduler tick.
        scheduler_state=resume['scheduler']
        previous_lrs=scheduler_state.get('_last_lr',scheduler_state.get('last_lr'))
        if previous_lrs is None:
            raise KeyError('Checkpoint scheduler state has neither _last_lr nor last_lr')
        if len(previous_lrs)!=len(opt.param_groups):
            raise ValueError(f'Checkpoint LR groups ({len(previous_lrs)}) do not match optimizer groups ({len(opt.param_groups)})')
        for group,lr in zip(opt.param_groups,previous_lrs):group['lr']=lr
    else:
        scheduler=build_cosine_scheduler(opt,warmup_steps=cfg.warmup,total_steps=cfg.lr_decay_steps,min_lr_ratio=cfg.min_lr_ratio)
        if resume:
            scheduler.load_state_dict(resume['scheduler']);assert scheduler.last_epoch==start
    args=training_args(cfg,run);args.ckpt_every=1000;args.epoch_steps=1000
    dataset,sampler,_,_,_=loader(cfg,args,log,0)
    bank=StdBank(cfg);process=diffusion(cfg)
    save(run/'resolved_config.json',spec['config']);save(run/'identity.json',spec['identity'])
    if resume:assert restore_rng(resume['rng'],log)
    else:seed(cfg.seed+10000)
    forward=torch.compile(raw,mode='default') if cfg.compile else raw
    log.info('Training initialization=%s compile=%s batch=%s microbatch=%s; no gradient checkpointing',
             spec.get('initialization','source'),cfg.compile,cfg.batch,cfg.microbatch or cfg.batch)
    raw.train();metrics=open(run/'train.jsonl','a',buffering=1)
    stop_requested=False
    def request_stop(*_):
        nonlocal stop_requested
        stop_requested=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    begun=time.monotonic();checkpoint=None;step_number=start
    milestones=set(spec['milestones'])|{a.stop}
    while step_number<a.stop and not stop_requested:
        epoch=step_number//1000;offset=(step_number%1000)*cfg.batch
        gen=torch.Generator().manual_seed(cfg.seed+epoch)
        batches=DataLoader(dataset,batch_size=cfg.batch,sampler=OffsetSampler(sampler,epoch,offset),
            num_workers=cfg.workers,pin_memory=True,drop_last=True,generator=gen)
        for batch in batches:
            if step_number>=a.stop or stop_requested:break
            z=batch['latent'].cuda(non_blocking=True).float().to(memory_format=torch.channels_last)
            clean=batch['std_soft'].cuda(non_blocking=True).float()
            if batch['std_soft'].dtype==torch.uint8:clean=clean/255.
            if clean.ndim==3:clean=clean[:,None]
            g=bank(batch,'cuda',step_number,clean=clean)
            w=batch['y_callig_raw'].cuda(non_blocking=True);s=batch['y_script'].cuda(non_blocking=True)
            opt.zero_grad(set_to_none=True)
            if auxiliary:
                loss,conditions=step(forward,process,z,g,w,s,cfg.microbatch or cfg.batch,return_conditions=True)
                aux_stats=auxiliary.backward(forward,z,g,conditions,step_number,
                    img_ids=batch['img_id'].tolist(),reader=reader)
                decoder_norm=auxiliary.clip_decoder()
            else:loss=step(forward,process,z,g,w,s,cfg.microbatch or cfg.batch)
            norm=torch.nn.utils.clip_grad_norm_(raw.parameters(),1.,error_if_nonfinite=True)
            opt.step();scheduler.step();step_number+=1;update_ema(ema,raw,.999)
            if auxiliary:auxiliary.update_ema()
            if step_number%50==0 or step_number==1:
                rec=dict(step=step_number,fm=float(loss),grad_norm=float(norm),clipped=float(norm)>1,
                    step_s=(step_number-start)/(time.monotonic()-begun),lr=scheduler.get_last_lr(),
                    peak_GiB=torch.cuda.max_memory_allocated()/2**30)
                if auxiliary:rec.update(auxiliary=aux_stats,decoder_grad_norm=decoder_norm)
                metrics.write(json.dumps(rec)+'\n');print(json.dumps(rec),flush=True)
            if step_number in milestones or step_number in (100,500):
                rng=capture_rng()
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    diagnostics=condition_report(raw,cfg)
                if cfg.style_matrix_repr>=0:diagnostics['gradient']=gradient_report(raw)
                save(run/'diagnostics'/f'{step_number:07d}.json',diagnostics)
                assert restore_rng(rng)
            if step_number in milestones or step_number%cfg.ckpt_every==0:
                ck=build_checkpoint(raw,opt,ema,scheduler,args,step_number,log)
                ck['matrix_identity']=spec['identity'];ck['rng_meta']['resume_kind']='deterministic_pair_sampler_offset'
                if auxiliary:ck.update(auxiliary.checkpoint())
                checkpoint=run/'checkpoints'/f'{step_number:07d}.pt'
                checkpoint.parent.mkdir(exist_ok=True)
                WRITER.submit(ck,str(checkpoint))
        del batches
    if checkpoint is None or checkpoint.stem!=f'{step_number:07d}':
        checkpoint=run/'checkpoints'/f'{step_number:07d}.pt';checkpoint.parent.mkdir(exist_ok=True)
        ck=build_checkpoint(raw,opt,ema,scheduler,args,step_number,log);ck['matrix_identity']=spec['identity']
        if auxiliary:ck.update(auxiliary.checkpoint())
        WRITER.submit(ck,str(checkpoint))
    drain_ckpt(log,strict=True)
    save(run/'completion.json',dict(step=step_number,requested=a.stop,stopped=stop_requested,
        checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
        seconds=time.monotonic()-begun))
    metrics.close()
    if stop_requested:raise SystemExit(75)


if __name__=='__main__':main()
