"""Actual source weights, backward, Adam and EMA; one isolated arm per process."""
import argparse
import copy
import json
from pathlib import Path
import time
import subprocess
import threading
import math
import torch
from src.unet.io import diffusion
from src.train.ema import update_ema
from src.unet.style.diagnostics import inspect,gradient_report
from ..runtime.common import construct,optimizer,synthetic,step,save


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--out',required=True)
    p.add_argument('--micro',type=int,default=256);p.add_argument('--steps',type=int,default=25)
    p.add_argument('--capacity-only',action='store_true',help='Synthetic graph capacity, no source/data/table dependency')
    p.add_argument('--real-data',action='store_true',help='Short end-to-end smoke, no checkpoints')
    p.add_argument('--data-config',default='assets/u100_p1p2_20261025/p2.json')
    p.add_argument('--source',default='')
    a=p.parse_args();torch.set_num_threads(8);torch.backends.cudnn.benchmark=True
    spec=json.loads(Path(a.spec).read_text())
    if a.real_data:
        if a.capacity_only:p.error('real-data and capacity-only are exclusive')
        base=json.loads(Path(a.data_config).read_text());base.update(spec['config']);spec['config']=base
    if a.source:spec['source']=a.source
    model,cfg,transfer=construct(spec,initialize_source=not a.capacity_only)
    if a.capacity_only:
        # Mature-shaped nonzero paths allow every style gradient to be checked.
        # This is a disposable memory probe, not a scientifically initialized run.
        from src.unet.blocks import ResBlock
        for layer in model.backbone.modules():
            if isinstance(layer,ResBlock):torch.nn.init.normal_(layer.conv2.weight,std=.001)
        torch.nn.init.normal_(model.backbone.output[-1].weight,std=.001)
    if not 1<=a.micro<=cfg.batch:raise ValueError('Invalid microbatch')
    assert not any(getattr(m,'gradient_checkpointing',False) for m in model.modules())
    probe=synthetic(cfg)
    with torch.autocast('cuda',dtype=torch.bfloat16):diagnostics=inspect(model,probe,calibrate=True)
    ema=copy.deepcopy(model).eval().requires_grad_(False);opt=optimizer(model,cfg)
    process=diffusion(cfg);model.train()
    inputs=synthetic(cfg,cfg.batch);z,_,g,w,s=inputs
    stream=None
    if a.real_data:
        import logging
        from src.unet.io import training_args
        from src.unet.data import loader,StdBank
        args=training_args(cfg,Path(a.out).parent)
        dataset,sampler,batches,_,_=loader(cfg,args,logging.getLogger('data-smoke'),0)
        bank=StdBank(cfg);stream=iter(batches)
    forward=torch.compile(model) if cfg.compile else model
    torch.cuda.reset_peak_memory_stats();wall=time.monotonic();begin=None
    gradient=None;timings=[];optim_times=[];ema_times=[];utilization=[];finished=threading.Event()
    def sample_utilization():
        while not finished.is_set():
            value=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu',
                '--format=csv,noheader,nounits'],text=True)
            utilization.append(int(value.strip()));finished.wait(2)
    for i in range(a.steps+5):
        if i==5:
            torch.cuda.synchronize();begin=time.monotonic()
            threading.Thread(target=sample_utilization,daemon=True).start()
        if stream is not None:
            batch=next(stream)
            z=batch['latent'].cuda(non_blocking=True).float().to(memory_format=torch.channels_last)
            clean=batch['std_soft'].cuda(non_blocking=True).float()
            if batch['std_soft'].dtype==torch.uint8:clean=clean/255.
            if clean.ndim==3:clean=clean[:,None]
            g=bank(batch,'cuda',i,clean=clean)
            w=batch['y_callig_raw'].cuda(non_blocking=True);s=batch['y_script'].cuda(non_blocking=True)
        opt.zero_grad(set_to_none=True);loss=step(forward,process,z,g,w,s,a.micro,timing=timings if i>=5 else None)
        if i==0:gradient=gradient_report(model)
        e=[torch.cuda.Event(enable_timing=True) for _ in range(3)]
        if i>=5:e[0].record()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        opt.step()
        if i>=5:e[1].record()
        update_ema(ema,model,.999)
        if i>=5:e[2].record();optim_times.append(e[:2]);ema_times.append(e[1:])
    torch.cuda.synchronize();elapsed=time.monotonic()-begin
    finished.set()
    report=dict(arm=spec['arm'],machine=spec['machine'],global_batch=cfg.batch,microbatch=a.micro,
        accum=math.ceil(cfg.batch/a.micro),microbatch_sizes=[min(a.micro,cfg.batch-j) for j in range(0,cfg.batch,a.micro)],
        gradient_checkpointing=False,step_s=a.steps/elapsed,measured_steps=a.steps,elapsed=elapsed,
        startup_seconds=begin-wall,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
        peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,compile=cfg.compile,
        parameters={key:sum(p.numel() for n,p in model.named_parameters() if n.startswith(key+'.'))
                    for key in ('backbone','style_representation','style_injections')},
        diagnostics=diagnostics,gradient=gradient,transfer=transfer,capacity_only=a.capacity_only,
        real_data_smoke=a.real_data,scientific_checkpoint_saved=False,
        final_loss=float(loss),final_grad=float(norm),torch_version=torch.__version__,samples_s=cfg.batch*a.steps/elapsed,
        gpu_utilization_mean=sum(utilization)/len(utilization) if utilization else None,
        gpu_ms=dict(forward=sum(v[0].elapsed_time(v[1]) for v in timings)/a.steps,
                    backward=sum(v[1].elapsed_time(v[2]) for v in timings)/a.steps,
                    clip_optimizer=sum(v[0].elapsed_time(v[1]) for v in optim_times)/a.steps,
                    ema=sum(v[0].elapsed_time(v[1]) for v in ema_times)/a.steps))
    save(a.out,report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
