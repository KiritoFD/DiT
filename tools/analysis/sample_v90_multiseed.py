"""Canonical final-checkpoint three-seed samples and content metrics, no training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKL

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.eval.model_io import load_model_from_ckpt
from src.eval.in_mem_eval import _get_cache, compute_set_metrics
from src.eval.sampler import build_diffusion, sample_latents
from src.eval.decoder import _lpips_per_sample


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    model,args=load_model_from_ckpt(a.checkpoint,device='cuda',use_ema=True)
    def sha(path):
        h=hashlib.sha256()
        with open(path,'rb') as f:
            for block in iter(lambda:f.read(2**20),b''):h.update(block)
        return h.hexdigest()
    (out/'manifest.json').write_text(json.dumps({'checkpoint':str(Path(a.checkpoint).resolve()),
        'checkpoint_sha256':sha(a.checkpoint),'weights':'ema','seeds':[0,1,2],
        'sampler':'heun','steps':50,'cfg':1.,'sde_gamma':0.,'vae_sf':.18215,'vae_shift':0.,
        'std_soft_sigma':.65,'csv_sha256':sha('exp-std/csv/eval200_fixed.csv'),
        'lpips':'VGG, canonical native evaluator'},indent=2))
    args.std_soft_root='exp-std/data/std_soft_glyphs_v1';args.std_soft_eval_sigma=.65
    cache=_get_cache('exp-std/csv/eval200_fixed.csv',187,None,'',args)
    vae=AutoencoderKL.from_pretrained('data/pretrained/sd-vae-ft-ema').cuda().eval()
    diff=build_diffusion(50,diffusion_type='flow',flow_kwargs={'sampler':'heun','sde_gamma':0.,'shift':1.})
    gt=(cache['gts'].cpu()+1)/2;gt_np=gt.numpy().transpose(0,2,3,1)
    results={}
    for seed in (0,1,2):
        noise=torch.randn((187,4,32,32),generator=torch.Generator().manual_seed(seed))
        z=sample_latents(model,diff,noise,cache['conds'],1.,11,torch.device('cuda'),
            skel=cache['skels_latent'],seed=seed,hier_conds=cache['hier_conds'])
        pred=torch.empty_like(gt)
        for i in range(0,187,8):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                x=vae.decode(z[i:i+8].cuda()/.18215).sample
            pred[i:i+8]=((x.float().clamp(-1,1)+1)/2).cpu()
        pp=pred.numpy().transpose(0,2,3,1)
        metrics=compute_set_metrics(pp,gt_np,187)
        lp=_lpips_per_sample(pp,gt_np,enabled=True)
        if lp is None or len(lp)!=187 or not np.isfinite(lp).all():
            raise RuntimeError('Missing valid LPIPS-VGG; refuse a falsely complete evaluation')
        dest=out/f'seed{seed}';dest.mkdir()
        for i in range(187):Image.fromarray((pp[i]*255).astype(np.uint8)).save(dest/f'g{i}.png')
        results[str(seed)]={'ssim':metrics.ssims,'ink_iou':metrics.inkious,'frag':metrics.frags,
            'near_blank':metrics.nb,'lpips_vgg':list(map(float,lp)),
            'noise_sha256':hashlib.sha256(noise.numpy().tobytes()).hexdigest()}
        (out/'content.json').write_text(json.dumps(results),encoding='utf-8')
        print('MULTISEED_COMPLETE',seed,flush=True)


if __name__=='__main__':main()
