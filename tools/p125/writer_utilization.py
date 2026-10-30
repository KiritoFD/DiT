"""Exp A: writer utilization across the P75 chain (no training).

(1) FM-loss gain by timestep: dL(t) = L_FM(w_wrong,t) - L_FM(w_correct,t) over
    t in {0.15,0.35,0.55,0.75,0.90} on real Top10 latents.
(2) Per-layer ablation: zero each style-injection scale's gain in turn and
    report how much of the writer gain each scale carries.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

from tools.p125.writer_swap import SCRIPT_SLOT


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--ckpt',default='')  # unused in chain mode
    p.add_argument('--train-csv',default='assets/u100_p1p2_20261025/p2_train.csv')
    p.add_argument('--n',type=int,default=128)
    p.add_argument('--out',default='docs/1031/writer_utilization.json')
    p.add_argument('--ckpts',nargs='+',default=[
        'assets/p125_20261030/execution/runs/P0/checkpoints/0050000.pt',
        'assets/p125_20261030/execution/runs/P1/checkpoints/0005000.pt',
        'assets/p125_20261030/execution/runs/P2/checkpoints/0015000.pt',
        'assets/p125_20261030/execution/runs/P3A/checkpoints/0020000.pt'])
    a=p.parse_args()
    from src.unet.io import load
    from diffusers import AutoencoderKL
    from PIL import Image
    vae=AutoencoderKL.from_pretrained('pretrained_models/sd-vae-ft-ema',local_files_only=True).cuda().eval().requires_grad_(False)
    sf=cfg.vae_scaling_factor;shift=cfg.vae_shift_factor
    zs=[];ws=[];ss=[]
    with torch.no_grad():
        for i in pick:
            r=rows[i]
            img=torch.from_numpy(np.asarray(Image.open(r['image_path']).convert('RGB')
                .resize((256,256),Image.Resampling.BICUBIC)).copy()).permute(2,0,1).float()/255.
            z=vae.encode(img[None].cuda()).latent_dist.mode()
            zs.append(((z-shift)*sf)[0])
            ws.append(int(r['calligrapher_id']));ss.append(int(r['script_id']))
    z=torch.stack(zs)
    by_script=defaultdict(set)
    for w,s2 in zip(ws,ss):by_script[s2].add(w)
    wrong_ws=[]
    for i,(w,s2) in enumerate(zip(ws,ss)):
        others=sorted(by_script[s2]-{w})
        wrong_ws.append(others[hash(str(w))%len(others)])
    del vae
    ts=[0.15,0.35,0.55,0.75,0.90]
    eps=torch.randn_like(z)
    def fm_loss(model,w_arr,t):
        tt=torch.full((len(z),),t,device='cuda')
        zt=(1-tt[:,None,None,None])*z+tt[:,None,None,None]*eps
        with torch.autocast('cuda',dtype=torch.bfloat16):
            v=model(zt,tt*1000.,torch.tensor(w_arr,device='cuda'),g=torch.randn(len(z),1,64,64,device='cuda'),
                y_script=torch.tensor(ss,device='cuda'),condition_dropped=True)
        target=eps-z
        return ((v.float()-target)**2).mean(-1).sum(-1)
    result={'per_checkpoint':{}}
    for ckpt_path in a.ckpts:
        tag=Path(ckpt_path).parents[1].name
        model,cfg2,_=load(ckpt_path,'cuda',use_ema=True);model.eval()
        per_t={}
        for t in ts:
            l_c=fm_loss(model,ws,t).detach()
            l_w=fm_loss(model,wrong_ws,t).detach()
            l_n=fm_loss(model,[47]*len(ws),t).detach()
            per_t[str(t)]=dict(writer_gain=round(float((l_w-l_c).mean()),5),
                null_gain=round(float((l_n-l_c).mean()),5),
                writer_gain_positive_share=round(float(((l_w-l_c)>0).float().mean()),3))
        result['per_checkpoint'][tag]=per_t
        print(tag,json.dumps(per_t),flush=True)
        del model;torch.cuda.empty_cache()
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.out).write_text(json.dumps(result,indent=1))
    print('WROTE',a.out)


if __name__=='__main__':main()
