"""Per-arm val100 panel generation (seed 0/1) for the pilot comparison."""
import argparse
import csv
from pathlib import Path
import numpy as np
import torch
from .common import load_s5,load_vae_decoder,TMP_ROOT
from src.unet.data import StdSoftGlyphStore
from .gen_swap import heun_sample


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--arm',required=True)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--csv',default='assets/u100_pretrain_20261024/validation_val100.csv')
    a=p.parse_args()
    out=TMP_ROOT/'valgen'/f'{a.arm}_seed{a.seed}';out.mkdir(parents=True,exist_ok=True)
    model,cfg,ck=load_s5()
    if a.arm!='S5_START':
        ckpt=TMP_ROOT/f'ckpt_{a.arm}'/f'{a.arm}_{5000 if not a.arm.startswith("D") else 2000:07d}.pt'
        model.load_state_dict(torch.load(ckpt,map_location='cpu')['delta'],strict=True)
    model.eval()
    for prm in model.parameters():prm.requires_grad_(False)
    vae=load_vae_decoder()
    rows=list(csv.DictReader(open(a.csv,encoding='utf-8-sig')))
    store=StdSoftGlyphStore('exp-std/data/std_soft_glyphs_v1',sigma_min=.65,sigma_max=.65,eval_sigma=.65)
    from .gen_swap import heun_sample
    from src.unet.io import diffusion as _diff
    from PIL import Image
    proc=_diff(cfg)
    done=0;B=25
    for start in range(0,len(rows),B):
        chunk=rows[start:start+B]
        gs=[];ws=[];ss=[]
        for r in chunk:
            try:
                g=store.get(int(r['script_id']),r['character'],sigma=.65).cuda().float()
                if g.ndim==2:g=g[None]
            except Exception:
                g=torch.zeros(1,64,64).cuda()
            gs.append(g);ws.append(int(r['calligrapher_id']));ss.append(int(r['script_id']))
        g=torch.stack(gs)
        seed_offset=(1000 if a.seed==1 else 0)
        noise=torch.stack([torch.randn(4,32,32,generator=torch.Generator().manual_seed(
            int(float(r['eval_noise_seed'] or 0))+seed_offset)) for r in chunk]).cuda()
        w=torch.tensor(ws,device='cuda');s=torch.tensor(ss,device='cuda')
        with torch.no_grad():
            z=heun_sample(model,proc,noise,w,g,s)
            img=((vae.decode(z.float()/0.18215).sample+1.)/2.).clamp(0,1)
        for i,r in enumerate(chunk):
            Image.fromarray((img[i].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)).save(out/f'g{start+i}.png')
        done+=len(chunk)
        print('valgen',a.arm,a.seed,done,flush=True)
    print('VALGEN-DONE',a.arm,a.seed,flush=True)


if __name__=='__main__':main()
