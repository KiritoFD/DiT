"""Full J0--J3 disposable optimizer steps, including VAE/LPIPS/decoder Adam+EMA."""
import argparse
import copy
import json
from pathlib import Path
import time
import torch
from src.unet.io import Config,build,diffusion
from src.unet.blocks import ResBlock
from src.train.ema import update_ema
from src.train.step import build_cosine_scheduler
from src.unet.joint.objective import JointDecoderObjective
from ..runtime.common import optimizer,synthetic,step,save,seed


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['J0','J1','J2','J3'],required=True)
    p.add_argument('--fm-micro',type=int,default=448);p.add_argument('--image-micro',type=int,default=16)
    p.add_argument('--image-batch',type=int,default=16);p.add_argument('--steps',type=int,default=30)
    p.add_argument('--out',required=True)
    a=p.parse_args();torch.set_num_threads(8);torch.backends.cudnn.benchmark=True;seed(90)
    cfg=Config(channels=(192,384,576),num_res_blocks=4,num_writers=259,num_scripts=3,
        freeze_writer_table=False,batch=512,max_steps=20000,lr_decay_steps=20000,drop_writer=.1)
    model=build(cfg).cuda().to(memory_format=torch.channels_last)
    for layer in model.modules():
        if isinstance(layer,ResBlock):torch.nn.init.normal_(layer.conv2.weight,std=.001)
    torch.nn.init.normal_(model.output[-1].weight,std=.001)
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    objective=JointDecoderObjective(a.arm,cfg,image_micro=a.image_micro,image_batch=a.image_batch)
    opt=optimizer(model,cfg,objective.vae.decoder)
    scheduler=build_cosine_scheduler(opt,warmup_steps=300,total_steps=20000,min_lr_ratio=.1)
    model.train();z,_,g,w,s=synthetic(cfg,512)
    truth=torch.rand(a.image_batch,3,256,256,device='cuda')*2-1
    begin=None;startup=time.monotonic();events=[];gradient={}
    torch.cuda.reset_peak_memory_stats()
    for i in range(a.steps+3):
        if i==3:torch.cuda.synchronize();begin=time.monotonic()
        e=[torch.cuda.Event(enable_timing=True) for _ in range(4)];e[0].record()
        opt.zero_grad(set_to_none=True)
        loss,conditions=step(model,diffusion(cfg),z,g,w,s,a.fm_micro,return_conditions=True)
        e[1].record()
        if i==0:before=model.output[-1].weight.grad.detach().clone()
        stats=objective.backward(model,z,g,conditions,i,targets=truth)
        e[2].record()
        if i==0:
            gradient['unet_image_delta']=float((model.output[-1].weight.grad-before).float().norm())
            d=objective.vae.decoder.conv_out.weight.grad
            gradient['decoder_output_grad']=float(d.norm()) if d is not None else 0.
            assert (gradient['unet_image_delta']>1e-9)==objective.image,gradient
            assert (gradient['decoder_output_grad']>1e-9)==objective.reconstruction,gradient
        unet_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        decoder_norm=objective.clip_decoder()  # Separate clip prevents spurious J2->U-Net coupling.
        opt.step();scheduler.step();update_ema(ema,model,.999);objective.update_ema()
        e[3].record()
        if i>=3:events.append(e)
    torch.cuda.synchronize();elapsed=time.monotonic()-begin
    objective.checkpoint()  # Check frozen encoder/quant/post state, never save probe weights.
    report=dict(arm=a.arm,global_batch=512,fm_microbatch=a.fm_micro,
        fm_microbatch_sizes=[min(a.fm_micro,512-j) for j in range(0,512,a.fm_micro)],
        image_batch=a.image_batch,image_microbatch=a.image_micro,optimizer_steps=a.steps,
        step_s=a.steps/elapsed,samples_s=512*a.steps/elapsed,startup_seconds=begin-startup,
        peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
        peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,gradient_checkpointing=False,
        image_gradient=gradient,unet_grad_norm=float(unet_norm),decoder_grad_norm=decoder_norm,
        frozen_encoder_unchanged=True,capacity_only=True,scientific_checkpoint_saved=False,
        loss_weights_for_capacity_only=dict(image=.1,reconstruction=1.,l1=1.,lpips=.1,edge=.1),
        gpu_ms=dict(fm=sum(e[0].elapsed_time(e[1]) for e in events)/a.steps,
                    image_and_reconstruction=sum(e[1].elapsed_time(e[2]) for e in events)/a.steps,
                    clip_optimizer_scheduler_ema=sum(e[2].elapsed_time(e[3]) for e in events)/a.steps),
        last_aux=stats,torch_version=torch.__version__)
    save(a.out,report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
