"""实验2: locate S5's writer pathway without training.

A) Per-sample FM loss deltas: correct vs wrong vs null writer at
   t∈{0.15,0.35,0.55,0.75,0.90}, identical noise (training_losses accepts noise=).
B) Causal activation splicing: replace the StyleState fields/tokens for
   scales {32,16} (or all) with null-writer / wrong-writer responses, keeping
   the backbone intact; measure FM delta, E-G margin delta, LPIPS vs full.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .common import EGEncoder,load_s5,load_vae_decoder,decode,save_json,PILOT_ROOT
from src.unet.data import loader,StdBank
from src.unet.io import training_args,diffusion
from src.unet.style.representations import StyleState

TIMES=(0.15,0.35,0.55,0.75,0.90)
SCALES=(32,16,8)  # pyramid order used by MatrixUNet


def splice_state(base,other,scales):
    """StyleState with `other`'s fields/tokens at the given scales."""
    fields=list(base.fields)
    scale_to_idx={s:i for i,s in enumerate(SCALES)}
    for s in scales:
        fields[scale_to_idx[s]]=other.fields[scale_to_idx[s]]
    return StyleState(base.global_feature,other.tokens,tuple(fields),base.valid)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--batches',type=int,default=24)
    p.add_argument('--out',default=PILOT_ROOT/'writer_path_ablation.csv')
    a=p.parse_args()
    torch.manual_seed(21)
    model,cfg,ck=load_s5()
    model.eval()
    for prm in model.parameters():prm.requires_grad_(False)
    vae=load_vae_decoder()
    enc=EGEncoder()
    args=training_args(cfg,PILOT_ROOT/'tmp_runs')
    dataset,sampler,_,_,_=loader(cfg,args,None,0)
    process=diffusion(cfg)
    from torch.utils.data import DataLoader
    it=iter(DataLoader(dataset,batch_size=64,num_workers=4))
    base_rep=model.style_representation
    rows=[]
    done=0
    while done<a.batches:
        try:batch=next(it)
        except StopIteration:it=iter(DataLoader(dataset,batch_size=64,num_workers=4));batch=next(it)
        z=batch['latent'].cuda().float()
        g=batch['std_soft'].cuda().float()
        if g.dtype==torch.uint8:g=g/255.
        if g.ndim==3:g=g[:,None]
        w=batch['y_callig'].cuda()
        s=batch['y_script'].cuda()
        writers=w.tolist()
        for t_val in TIMES:
            t=torch.full((len(z),),t_val,device='cuda')
            noise=torch.randn_like(z)
            kw=lambda ww:dict(g=g,y_callig=ww,y_script=s,condition_dropped=True)
            with torch.no_grad():
                l_c=process.training_losses(model,z,t,kw(w),noise=noise)['loss']
                l_w=process.training_losses(model,z,t,kw((w+3)%10),noise=noise)['loss']
                l_n=process.training_losses(model,z,t,kw(torch.full_like(w,10)),noise=noise)['loss']
            for name,d in (('wrong',l_w-l_c),('null',l_n-l_c)):
                rows.append(dict(part='fm_delta',config=name,t=t_val,n=len(d),
                    mean=round(float(d.mean()),5),pct_pos=round(100*float((d>0).float().mean()),1),
                    writer_breakdown=json.dumps({str(ww):round(float(d[[i for i,x in enumerate(writers) if x==ww]].mean()),4) for ww in sorted(set(writers))})))
            # splicing on the first 8 samples of the batch (decode cost)
            nb=min(8,len(z))
            with torch.no_grad():
                glyph=g[:nb]
                st_c=base_rep(w[:nb],None,glyph) if False else None
            # build pyramid exactly as MatrixUNet.forward does
            base=model.backbone
            with torch.no_grad():
                pyr=[base.std_stem(glyph)]
                for down in base.std_down:pyr.append(base.act(down(pyr[-1])))
                st_c=base_rep(w[:nb],pyr,glyph)
                st_n=base_rep(torch.full_like(w[:nb],10),pyr,glyph)
                st_w=base_rep((w[:nb]+3)%10,pyr,glyph)
                for config in ('full','null_s0','null_s1','null_both','wrong'):
                    if config=='full':st=st_c
                    elif config=='null_s0':st=splice_state(st_c,st_n,[32])
                    elif config=='null_s1':st=splice_state(st_c,st_n,[16])
                    elif config=='null_both':st=splice_state(st_c,st_n,[32,16])
                    else:st=st_w
                    model.style_representation.forward=lambda w,pyr,g,_st=st:_st
                    tt=torch.full((nb,),t_val,device='cuda')
                    loss=process.training_losses(model,z[:nb],tt,
                        dict(g=glyph,y_callig=w[:nb],y_script=s[:nb],condition_dropped=True),noise=noise[:nb])['loss']
                    v=model(z[:nb],tt*1000.,w[:nb],g=glyph,y_script=s[:nb],condition_dropped=True)
                    z0=z[:nb]-tt[:,None,None,None]*v.float()
                    img=decode(vae,z0)
                    eg=enc.features(img)
                    d_eg=float(eg.norm(dim=1).mean())  # margin needs bank; keep norm here
                    rows.append(dict(part='splice',config=config,t=t_val,n=nb,
                        mean=round(float(loss.mean()),5),pct_pos='',
                        writer_breakdown=json.dumps(dict(d_eg_norm=round(d_eg,4)))))
                model.style_representation=base_rep
        done+=1
        print('batch',done,flush=True)
    with open(a.out,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['part','config','t','n','mean','pct_pos','writer_breakdown'])
        w.writeheader();w.writerows(rows)
    # quick console summary
    agg=defaultdict(list)
    for r in rows:
        if r['part']=='fm_delta':agg[(r['config'],r['t'])].append(r['mean'])
    for k in sorted(agg):print(k,round(float(np.mean(agg[k])),5))


if __name__=='__main__':main()
