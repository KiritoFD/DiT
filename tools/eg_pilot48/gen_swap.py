"""统一评测生成: Writer-Swap panel + per-arm val100 generation.

Swap: valid slots × 10 chars (from the 187 heldout protocol) × 2 seeds ×
{correct, wrong, null} writer conditions, heun-50 sampling, decoder decode.
Saved PNGs feed E-G/E-G2 direction/amplitude and content-stability metrics.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from .common import load_s5,load_vae_decoder,PILOT_ROOT,TMP_ROOT,sha256_file
from src.unet.data import StdSoftGlyphStore


def heun_sample(model,process,noise,w,g,s,steps=50):
    """Integrate the FM ODE from t=1 (noise) to t=0 (data), heun."""
    x=noise.clone()
    ts=np.linspace(1.,0.,steps+1)
    for i in range(steps):
        t=torch.full((len(x),),ts[i],device=x.device)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            v=model(x,t*1000.,w,g=g,y_script=s,condition_dropped=True).float()
        x_mid=x+(ts[i+1]-ts[i])*v
        t2=torch.full((len(x),),ts[i+1],device=x.device)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            v2=model(x_mid,t2*1000.,w,g=g,y_script=s,condition_dropped=True).float()
        x=x+(ts[i+1]-ts[i])*(v+v2)/2
    return x


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--ckpt',required=True,help='arm checkpoint path (or S5 start)')
    p.add_argument('--tag',required=True)
    p.add_argument('--chars-per-script',type=int,default=10)
    p.add_argument('--seeds',type=int,default=2)
    p.add_argument('--out',default=None)
    a=p.parse_args()
    out=Path(a.out or (TMP_ROOT/f'swap_{a.tag}'));out.mkdir(parents=True,exist_ok=True)
    model,cfg,ck=load_s5()
    if a.ckpt!='S5_START':
        model.load_state_dict(torch.load(a.ckpt,map_location='cpu')['delta'],strict=True)
    model.eval()
    for prm in model.parameters():prm.requires_grad_(False)
    vae=load_vae_decoder()
    import csv as _csv
    train_rows=list(_csv.DictReader(open(cfg.data_csv,encoding='utf-8-sig')))
    slots=defaultdict(set)
    for r in train_rows:slots[int(r['script_id'])].add(int(r['calligrapher_id']))
    held=list(_csv.DictReader(open('exp-std/csv/eval200_fixed.csv',encoding='utf-8-sig')))
    chars_by_script=defaultdict(list)
    for r in held:
        if r['character'] not in chars_by_script[int(r['script_id'])]:
            chars_by_script[int(r['script_id'])].append(r['character'])
    store=StdSoftGlyphStore('exp-std/data/std_soft_glyphs_v1',sigma_min=.65,sigma_max=.65,eval_sigma=.65)
    manifest=[]
    for script_id,writers in sorted(slots.items()):
        writers=sorted(writers)
        chars=chars_by_script.get(script_id,[])[:a.chars_per_script]
        if not chars or len(writers)<2:continue
        store_cache={}
        for ch in chars:
            try:store_cache[ch]=store.get(script_id,ch,sigma=.65).cuda().float()
            except Exception as e:print('glyph miss',script_id,ch,e,flush=True)
        chars=[c for c in chars if c in store_cache]
        for seed in range(a.seeds):
            gen=torch.Generator(device='cpu').manual_seed(1000+seed)
            for wi,w_correct in enumerate(writers):
                w_wrong=writers[(wi+1)%len(writers)]
                for cond,w_id in (('correct',w_correct),('wrong',w_wrong),('null',10)):
                    noise=torch.randn(len(chars),4,32,32,generator=gen).cuda()
                    w=torch.tensor([w_id]*len(chars),device='cuda')
                    s=torch.tensor([script_id]*len(chars),device='cuda')
                    g=torch.stack([store_cache[c] for c in chars])
                    if g.ndim==3:g=g[:,None]
                    with torch.no_grad():
                        z=heun_sample(model,process,noise,w,g,s)
                        img=((vae.decode(z.float()/0.18215).sample+1.)/2.).clamp(0,1)
                    for ci,ch in enumerate(chars):
                        fp=out/f'{script_id}_{w_correct}_{ch}_{seed}_{cond}.png'
                        from PIL import Image
                        Image.fromarray((img[ci].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)).save(fp)
                        manifest.append(dict(script=script_id,writer=w_correct,char=ch,seed=seed,cond=cond,
                            path=str(fp),gen_writer=w_id))
    save_json(out/'manifest.json',manifest)
    print('SWAP-DONE',len(manifest),'images',flush=True)


if __name__=='__main__':main()
