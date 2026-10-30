"""Exp E part 2: E-G gradient-transfer pilot WITHOUT touching the model.

Freeze S5@85k. Real Top10 latents + mid noise -> predicted clean latent zhat0.
Optimize zhat0 with the E-G same-script ranking loss (Adam on the latent,
60 steps). Before/after: E-G score, E-G2 score (independent verifier),
LPIPS and SSIM vs the real image, and the latent drift magnitude.
Decision rule: E-G up but E-G2 flat/negative -> cancel the training fork.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn

from tools.p125.eg2 import EG_MEAN,EG_STD


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--ckpt',default='assets/style_focus_85k_20261029/48_v3/runs/S5/checkpoints/0085000.pt')
    p.add_argument('--eg2',default='assets/p125_20261030/eg2.pt')
    p.add_argument('--refs',default='assets/p125_20261030/eg_refs.npz')
    p.add_argument('--csv',default='assets/u100_p1p2_20261025/p2_train.csv')
    p.add_argument('--n',type=int,default=24)
    p.add_argument('--steps',type=int,default=60)
    p.add_argument('--lr',type=float,default=2e-3)
    p.add_argument('--out',default='docs/1031/latent_pilot.json')
    a=p.parse_args()
    from src.unet.io import load
    from diffusers import AutoencoderKL
    from tools.eval_gen_with_disc import load_evaluator_model
    model,cfg,ck=load(a.ckpt,'cuda',use_ema=True);model.eval()
    vae=AutoencoderKL.from_pretrained(cfg.vae_path,local_files_only=True).cuda().eval().requires_grad_(False)
    sf=cfg.vae_scaling_factor;shift=cfg.vae_shift_factor
    eg,_kind,_=load_evaluator_model('assets/style_disc_ce_holdout.pt','cuda');eg.eval().requires_grad_(False)
    eg2_state=torch.load(a.eg2,map_location='cpu',weights_only=False)
    import torchvision
    eg2=torchvision.models.resnet34()
    import torch.nn as nn
    eg2.fc=nn.Linear(eg2.fc.in_features,10)
    eg2.load_state_dict(eg2_state['model']);eg2=eg2.cuda().eval().requires_grad_(False)
    def eg2_penult(images):
        m=eg2
        x=m.conv1(m.bn1(m.relu(m.maxpool(images))))
        x=m.layer1(x);x=m.layer2(x);x=m.layer3(x);x=m.layer4(x)
        return m.avgpool(x).flatten(1)
    refs=np.load(a.refs,allow_pickle=True)
    bank={k:torch.from_numpy(v['embs']).float().cuda() for k,v in refs.items()}
    rows=[r for r in csv.DictReader(open(a.csv,encoding='utf-8-sig'))
          if not (r.get('aug') or '').strip()]
    rng=np.random.default_rng(20261031)
    pick=list(rng.choice(len(rows),a.n,replace=False))
    def eg_score(im):
        x=F.interpolate(im,(160,160),mode='bilinear',align_corners=False)
        x=(x-EG_MEAN.to(x.device)[None])/EG_STD.to(x.device)[None]
        with torch.no_grad():
            e=eg.emb(x)
        e=e/e.norm(dim=-1,keepdim=True).clamp_min(1e-6)
        sims=[]
        for b in range(e.shape[0]):
            slot=f"{int(rows[pick[b]]['calligrapher_id'])}_{int(rows[pick[b]]['script_id'])}"
            sims.append(float(e[b:b+1]@bank[slot].T.topk(5,dim=1).values.mean()))
        return sims
    def eg2_score(im):
        x=F.interpolate(im,(128,128),mode='bilinear',align_corners=False)
        with torch.no_grad():
            e=eg2_penult(x)
        e=e/e.norm(dim=-1,keepdim=True).clamp_min(1e-6)
        out=[]
        for b in range(e.shape[0]):
            slot=f"{int(rows[pick[b]]['calligrapher_id'])}_{int(rows[pick[b]]['script_id'])}"
            out.append(float(e[b:b+1]@bank[slot].T.topk(5,dim=1).values.mean()))
        return out
    def content(im,r):
        with torch.no_grad():
            gt=vae.decode(torch.from_numpy(np.asarray(Image.open(r['image_path']).convert('RGB')
                .resize((256,256),Image.Resampling.BICUBIC)).copy())[None].permute(0,3,1,2).float().cuda()*2-1
                /1.).sample if False else None
        # GT is a real image, not a latent decode: compare in pixel space directly
        gt=np.asarray(Image.open(r['image_path']).convert('L').resize((256,256)),dtype=np.float32)/255
        g=np.asarray(F.interpolate(im,(256,256),mode='bilinear',align_corners=False)[0,0].float().cpu(),dtype=np.float32)/2+0.5
        s=ssim_fn(gt,np.clip(g,0,1),data_range=1.0)
        return float(s)
    samples=[]
    with torch.no_grad():
        for i in pick:
            r=rows[i]
            img=torch.from_numpy(np.asarray(Image.open(r['image_path']).convert('RGB')
                .resize((256,256),Image.Resampling.BICUBIC)).copy()).permute(2,0,1).float()/255.
            z=vae.encode(img[None].cuda()).latent_dist.mode()
            zs=((z-shift)*sf)[0]
            t=torch.tensor([0.5],device='cuda')
            eps=torch.randn_like(zs)
            zt=(1-t)*zs+t*eps
            with torch.autocast('cuda',dtype=torch.bfloat16):
                v=model(zt,t*1000.,torch.tensor([int(r['calligrapher_id'])],device='cuda'),
                    g=torch.randn(1,1,64,64,device='cuda'),
                    y_script=torch.tensor([{0:0,3:1,4:2}[int(r['script_id'])]],device='cuda'),
                    condition_dropped=True)
            zhat0=(zt-t*v).detach()
            slot=f"{int(r['calligrapher_id'])}_{int(r['script_id'])}"
            if slot not in bank:continue
            samples.append((r,zhat0,slot,img[0].cuda()))
    before_eg=[];before_eg2=[];before_ss=[]
    with torch.no_grad():
        for r,z0,slot,img in samples:
            im=vae.decode(z0[None]/sf+shift).sample
            before_eg.append(eg_score(im)[0])
            before_eg2.append(eg2_score(im)[0])
            gt=np.asarray(Image.open(r['image_path']).convert('L').resize((256,256)),dtype=np.float32)/255
            g=np.asarray(F.interpolate(im,(256,256))[0,0].float().cpu())/2+0.5
            before_ss.append(ssim_fn(gt,np.clip(g,0,1),data_range=1.0))
    zh=torch.cat([z[None] for _,z,_,_ in samples]).detach().clone().requires_grad_(True)
    adam=torch.optim.Adam([zh],lr=a.lr)
    slots=[sl for _,_,sl,_ in samples]
    for it in range(a.steps):
        im=vae.decode(zh/sf+shift).sample
        x=F.interpolate(im,(160,160),mode='bilinear',align_corners=False)
        x=(x-EG_MEAN.to(x.device)[None])/EG_STD.to(x.device)[None]
        e=eg.emb(x);e=e/e.norm(dim=-1,keepdim=True).clamp_min(1e-6)
        logits=torch.stack([e[b:b+1]@bank[sl].T.topk(5,dim=1).values.mean() for b,sl in enumerate(slots)])/0.07
        target=torch.tensor([sorted(bank).index(sl) for sl in slots],device='cuda')
        loss=F.cross_entropy(logits,target)
        adam.zero_grad();loss.backward();adam.step()
        if it%20==0:print('pilot step',it,'loss',round(float(loss),4),flush=True)
    after_eg=[];after_eg2=[];after_ss=[];drifts=[]
    with torch.no_grad():
        for b,(r,z0,slot,img) in enumerate(samples):
            im=vae.decode(zh[b:b+1]/sf+shift).sample
            after_eg.append(eg_score(im)[0])
            after_eg2.append(eg2_score(im)[0])
            gt=np.asarray(Image.open(r['image_path']).convert('L').resize((256,256)),dtype=np.float32)/255
            g=np.asarray(F.interpolate(im,(256,256))[0,0].float().cpu())/2+0.5
            after_ss.append(ssim_fn(gt,np.clip(g,0,1),data_range=1.0))
            drifts.append(float((zh[b]-z0).norm()/z0.norm().clamp_min(1e-6)))
    d_eg=[a-b for a,b in zip(after_eg,before_eg)]
    d_eg2=[a-b for a,b in zip(after_eg2,before_eg2)]
    d_ss=[a-b for a,b in zip(after_ss,before_ss)]
    out=dict(n=len(samples),steps=a.steps,lr=a.lr,
        mean_d_eg=round(float(np.mean(d_eg)),4),eg_improved_share=round(float(np.mean([x>0 for x in d_eg])),3),
        mean_d_eg2=round(float(np.mean(d_eg2)),4),eg2_improved_share=round(float(np.mean([x>0 for x in d_eg2])),3),
        mean_d_ssim=round(float(np.mean(d_ss)),4),ssim_regress_share=round(float(np.mean([x<-0.01 for x in d_ss])),3),
        latent_drift_median=round(float(np.median(drifts)),4),
        verdict=('EG-only' if float(np.mean(d_eg))>0 and float(np.mean(d_eg2))<=0 else
                 'EG-and-EG2' if float(np.mean(d_eg2))>0 else 'no-effect'))
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.out).write_text(json.dumps(out,indent=1))
    print(json.dumps(out))


if __name__=='__main__':main()
