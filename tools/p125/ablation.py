"""Exp D: causal ablation of S5 writer conditioning.

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
    p.add_argument('--ckpt',default='assets/style_focus_85k_20261029/48_v3/runs/S5/checkpoints/0085000.pt')
    p.add_argument('--train-csv',default='assets/u100_p1p2_20261025/p2_train.csv')
    p.add_argument('--shards',default='assets/generic48_kxl_top10_20261022/latents')
    p.add_argument('--n',type=int,default=128)
    p.add_argument('--out',default='docs/1031/ablation.json')
    a=p.parse_args()
    from src.unet.io import load
    from src.loss.flow_matching import FlowMatching
    model,cfg,ck=load(a.ckpt,'cuda',use_ema=True);model.eval()
    proc=FlowMatching(num_steps=50,sampler='heun',t_sampler='uniform')
    # real latents: VAE-encode a deterministic sample of Top10 GT images
    rows=[r for r in csv.DictReader(open(a.train_csv,encoding='utf-8-sig'))
          if not (r.get('aug') or '').strip()]
    rng=np.random.default_rng(31)
    pick=list(rng.choice(len(rows),a.n,replace=False))
    from diffusers import AutoencoderKL
    from PIL import Image
    vae=AutoencoderKL.from_pretrained(cfg.vae_path,local_files_only=True).cuda().eval().requires_grad_(False)
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
    wrong_ws=np.array([ws[i]+1 if (ws[i]+1)<47 else 10 for i in range(len(ws))])
    # same-script wrong writer: use another writer of the same script
    by_script=defaultdict(set)
    for w,s in zip(ws,ss):by_script[s].add(w)
    wrong_ws=[]
    for i,(w,s) in enumerate(zip(ws,ss)):
        others=sorted(by_script[s]-{w})
        wrong_ws.append(others[hash(str(w))%len(others)])
    del vae
    ts=[0.15,0.35,0.55,0.75,0.90]
    eps=torch.randn_like(z)
    def fm_loss(w_arr,t):
        tt=torch.full((len(z),),t,device='cuda')
        zt=(1-tt[:,None,None,None])*z+tt[:,None,None,None]*eps
        with torch.autocast('cuda',dtype=torch.bfloat16):
            v=model(zt,tt*1000.,torch.tensor(w_arr,device='cuda'),g=torch.randn(len(z),1,64,64,device='cuda'),
                y_script=torch.tensor(ss,device='cuda'),condition_dropped=True)
        target=eps-z
        return ((v.float()-target)**2).mean(-1).sum(-1)
    result={'per_t':{},'layers':{}}
    gains=defaultdict(dict)
    for t in ts:
        l_c=fm_loss(ws,t).detach()
        l_w=fm_loss(wrong_ws,t).detach()
        l_n=fm_loss([47]*len(ws),t).detach()
        gain=(l_w-l_c).mean().item()
        null_gap=(l_n-l_c).mean().item()
        result['per_t'][t]=dict(writer_gain=round(gain,5),null_gain=round(null_gap,5),
            per_writer={str(w):round(float((l_w-l_c)[i]),4) for i,w in enumerate(ws[:16])})
        gains[t]=(l_c,l_w,l_n)
    # per-layer ablation at t=0.55
    t=0.55
    base_c=fm_loss(ws,t);base_w=fm_loss(wrong_ws,t)
    base_gain=(base_w-base_c)
    for li in range(len(model.style_injections)):
        old=model.style_injections[li].gain.detach().clone()
        with torch.no_grad():model.style_injections[li].gain.zero_()
        c2=fm_loss(ws,t);w2=fm_loss(wrong_ws,t)
        with torch.no_grad():model.style_injections[li].gain.copy_(old)
        kept=float(((w2-c2)/base_gain.clamp_min(1e-9)).mean().item())
        result['layers'][li]=dict(writer_gain_retained=round(kept,4),
            scale=['16x16','32x32','64x64'][li] if li<3 else str(li))
    save=lambda:(Path(a.out).parent.mkdir(parents=True,exist_ok=True),Path(a.out).write_text(json.dumps(result,indent=1)))
    save();print(json.dumps(result['per_t'],indent=1));print(json.dumps(result['layers']))


if __name__=='__main__':main()
