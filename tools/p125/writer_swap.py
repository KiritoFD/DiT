"""Exp C: Writer-Swap panel (23 slots x 10 unseen chars x 2 seeds, CFG=1.0)
plus a CFG sweep {1.3,1.6,2.0} (seed0). Fixed STD/char/noise, only Writer varies.
Latents are decoded through the frozen VAE to 256px grayscale PNGs.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image

SCRIPT_SLOT={0:0,3:1,4:2}  # raw corpus script_id -> model script slot


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--ckpt',default='assets/style_focus_85k_20261029/48_v3/runs/S5/checkpoints/0085000.pt')
    p.add_argument('--train-csv',default='assets/u100_p1p2_20261025/p2_train.csv')
    p.add_argument('--generic-csv',default='assets/generic48_kxl_top10_20261022/train.csv')
    p.add_argument('--store',default='assets/generic48_kxl_top10_20261022/std_bbox64')
    p.add_argument('--out',default='assets/p125_20261030/swap_panel')
    p.add_argument('--cfgs',default='1.0,1.3,1.6,2.0')
    p.add_argument('--chars-per-script',type=int,default=10)
    p.add_argument('--seeds',type=int,default=2)
    a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    from src.unet.io import load,diffusion
    from diffusers import AutoencoderKL
    model,cfg,ck=load(a.ckpt,'cuda',use_ema=True)
    model.eval()
    proc=diffusion(cfg)
    vae=AutoencoderKL.from_pretrained(cfg.vae_path,local_files_only=True).cuda().eval().requires_grad_(False)
    sf=cfg.vae_scaling_factor;shift=cfg.vae_shift_factor
    # valid slots (>=200 imgs) from the Top10 training CSV
    rows=list(csv.DictReader(open(a.train_csv,encoding='utf-8-sig')))
    rows=[r for r in rows if not (r.get('aug') or '').strip()]
    cnt=defaultdict(int)
    for r in rows:cnt[(int(r['calligrapher_id']),int(r['script_id']))]+=1
    slots=sorted(s for s,c in cnt.items() if c>=200)
    train_keys={(int(r['script_id']),r['character']) for r in rows}
    gen_keys=set()
    for r in csv.DictReader(open(a.generic_csv,encoding='utf-8-sig')):
        gen_keys.add((int(r['script_id']),r['character']))
    # unseen chars per script: bank-covered, not in Top10/generic training
    store_rows=list(csv.DictReader(open(Path(a.store)/'index.csv',encoding='utf-8-sig')))
    by_script=defaultdict(list)
    for r in store_rows:
        key=(int(r['script_id']),r['character'])
        if key in train_keys or key in gen_keys:continue
        by_script[int(r['script_id'])].append(r)
    chars={}
    for script,rs in sorted(by_script.items()):
        rs=sorted(rs,key=lambda r:r['character'])
        chars[script]=rs[:a.chars_per_script]
    std_by_key={ (int(r['script_id']),r['character']):r for r in store_rows }
    writers_per_script=defaultdict(set)
    for (w,s) in slots:writers_per_script[s].add(w)
    manifest=[]
    cfgs=[float(x) for x in a.cfgs.split(',')]
    seed0=310310
    for script in sorted(chars):
        cs=chars[script]
        if len(cs)<a.chars_per_script:continue
        writers=sorted(writers_per_script[script])
        if not writers:continue
        std_batch=[]
        for r in cs:
            img=Image.open(Path(a.store)/r['path']).convert('L')
            std_batch.append(np.asarray(img,dtype=np.float32)/255.)
        g=torch.from_numpy(np.stack(std_batch))[:,None].cuda()
        for w in writers:
            for ci in range(len(cs)):
                for seed_i in range(a.seeds):
                    for c in cfgs:
                        only_main=(c==cfgs[0])
                        tag=f"w{w}_s{script}_c{ci}_seed{seed_i}" if only_main else \
                            f"cfg{c}_w{w}_s{script}_c{ci}_seed{seed_i}"
                        if only_main and seed_i>=a.seeds:continue
                        if not only_main and seed_i>=1:continue  # CFG sweep: seed0 only
                        fpath=out/f'{tag}.png'
                        if fpath.exists():manifest.append(dict(tag=tag,cfg=c,writer=w,script=script,
                            character=cs[ci]['character'],std_path=cs[ci]['path'],file=str(fpath)));continue
                        gm=torch.Generator(device='cuda').manual_seed(seed0+1000*seed_i+ci)
                        x_T=torch.randn(1,4,32,32,generator=gm,device='cuda')
                        kw=dict(g=g[ci:ci+1],y_callig=torch.tensor([w],device='cuda'),
                            y_script=torch.tensor([script],device='cuda'),condition_dropped=True,
                            cfg_scale=c)
                        with torch.autocast('cuda',dtype=torch.bfloat16):
                            sample=proc.sample(model,(1,4,32,32),x_T=x_T,model_kwargs=kw)
                        img=sample[0,0].float().cpu()
                        arr=((img+1)/2).clamp(0,1).numpy()
                        arr=(arr*255).astype(np.uint8)
                        Image.fromarray(arr).convert('L').save(fpath)
                        manifest.append(dict(tag=tag,cfg=c,writer=w,script=script,
                            character=cs[ci]['character'],std_path=cs[ci]['path'],file=str(fpath)))
        print('script',script,'done',flush=True)
    json.dump(manifest,open(out/'manifest.json','w'),ensure_ascii=False,indent=1)
    print('panel images:',len(manifest))


if __name__=='__main__':main()
