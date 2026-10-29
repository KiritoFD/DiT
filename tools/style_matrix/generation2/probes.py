"""Velocity mechanism audit and minimal CPU invariants, independent of training."""
import argparse
import csv
import json
from pathlib import Path
import torch
from tools.style_matrix.runtime.common import save


@torch.no_grad()
def velocity(checkpoint,protocol,out):
    from src.unet.io import load,eval_args
    from src.eval.in_mem_eval import _get_cache
    from diffusers import AutoencoderKL
    model,cfg,ck=load(checkpoint,'cuda');args=eval_args(cfg)
    rows=list(csv.DictReader(open(protocol,encoding='utf-8-sig')))
    cache=_get_cache(protocol,len(rows),cfg.img_root,None,args)
    n=min(len(rows),16);g=cache['skels_latent'][:n].cuda()
    w=torch.tensor([v[0] for v in cache['conds'][:n]],device='cuda');s=torch.tensor(cache['hier_conds'][2][:n],device='cuda')
    vae=AutoencoderKL.from_pretrained(cfg.vae_path,local_files_only=True).cuda().eval().requires_grad_(False)
    gt=cache['gts'][:n].cuda();z=(vae.encode(gt).latent_dist.mode()-cfg.vae_shift_factor)*cfg.vae_scaling_factor
    noise=cache['noise'][:n].cuda();results=[]
    for t in (.05,.25,.5,.75,1.):
        xt=(1-t)*z+t*noise;time=xt.new_full((n,),1000*t)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            content,style=model.components(xt,time,w,g,s,True)
            null=model.components(xt,time,torch.full_like(w,model.num_writers),g,s,True)
        if not torch.equal(content,null[0]) or torch.count_nonzero(null[1]):raise ValueError('Writer leaked into content/null')
        rms=lambda value:value.float().square().mean().sqrt()
        results.append(dict(t=t,style_rms=float(rms(style)),content_rms=float(rms(content)),
                            style_ratio=float(rms(style)/rms(content).clamp_min(1e-9)),null_exact=True))
    save(out,dict(checkpoint=str(checkpoint),protocol=str(protocol),time_bins=results))


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--csv',required=True)
    p.add_argument('--out',required=True);a=p.parse_args();velocity(a.checkpoint,a.csv,a.out)


if __name__=='__main__':main()
