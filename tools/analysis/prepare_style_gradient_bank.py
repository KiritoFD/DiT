"""Frozen v84A heldout bank: >=48 distinct glyphs x three actual noise seeds."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.eval.model_io import load_model_from_ckpt
from src.eval.in_mem_eval import _get_cache
from src.eval.sampler import sample_latents,build_diffusion


def select_characters(rows,slots,count=48,seed=89):
    rng=np.random.default_rng(seed);groups=defaultdict(list)
    for i,row in enumerate(rows):
        if (row['calligrapher'],row['script']) in slots:groups[row['calligrapher'],row['script']].append(i)
    for key in groups:rng.shuffle(groups[key])
    chosen=[];seen=set();keys=sorted(groups)
    while len(chosen)<count:
        progress=False
        for key in keys:
            while groups[key] and rows[groups[key][-1]]['character'] in seen:groups[key].pop()
            if groups[key]:
                i=groups[key].pop();chosen.append(i);seen.add(rows[i]['character']);progress=True
                if len(chosen)==count:break
        if not progress:raise ValueError('Insufficient distinct supported heldout characters')
    return chosen


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True)
    p.add_argument('--characters',type=int,default=48);p.add_argument('--seeds',default='0,1,2');p.add_argument('--steps',type=int,default=50);p.add_argument('--batch',type=int,default=8)
    # Freeze caller-relative paths before switching to the canonical repo cwd.
    a=p.parse_args();root=Path(a.root).resolve();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False);os.chdir(root)
    torch.set_num_threads(4);data=torch.load(root/'assets/writer_latent_data_20261016/data.pt',map_location='cpu',weights_only=False)
    csvfile=root/'exp-std/csv/eval200_fixed.csv';rows=list(csv.DictReader(open(csvfile,encoding='utf-8-sig')))
    selected=select_characters(rows,data['slots'],a.characters)
    ckpt=root/'assets/runs/v84A_std_cond/20261013-072215-v84A-std-cond/checkpoints/0200000.pt'
    model,args=load_model_from_ckpt(str(ckpt),device='cuda',use_ema=True);model.eval().requires_grad_(False)
    args.std_soft_root=str(root/'exp-std/data/std_soft_glyphs_v1');args.std_soft_eval_sigma=.65
    cache=_get_cache(str(csvfile),187,None,'',args)
    diffusion=build_diffusion(a.steps,diffusion_type='flow',flow_kwargs={'sampler':'heun','sde_gamma':0.})
    outputs=[];metadata=[];conds=[cache['conds'][i] for i in selected];h=tuple([[axis[i] for i in selected] for axis in cache['hier_conds']])
    skel=cache['skels_latent'][selected].cpu();seeds=list(map(int,a.seeds.split(',')))
    # No tail condition mutation: choose a divisor of the actual character count.
    batch=min(a.batch,a.characters)
    while a.characters%batch:batch-=1
    with torch.no_grad():
        for seed in seeds:
            noise=torch.randn((len(selected),4,32,32),generator=torch.Generator().manual_seed(seed))
            z=sample_latents(model,diffusion,noise,conds,1.,batch,torch.device('cuda'),skel=skel,seed=seed,hier_conds=h)
            outputs.append(z)
            for j,i in enumerate(selected):
                row=rows[i];metadata.append({'idx':len(metadata),'original_idx':i,'writer_id':int(conds[j][0]),
                    'writer':row['calligrapher'],'script':row['script'],'character':row['character'],'seed':seed,
                    'supported_slot':True,'gt_source_proxy':str(Path(row['image_path']).parent),
                    'noise_sha256':hashlib.sha256(noise[j].numpy().tobytes()).hexdigest(),
                    'g_sha256':hashlib.sha256(skel[j].numpy().tobytes()).hexdigest()})
            print('EXPANDED_BANK_SEED_COMPLETE',seed,len(z),flush=True)
    torch.save(torch.cat(outputs),out/'latents.pt')
    manifest={'checkpoint':str(ckpt),'checkpoint_sha256':hashlib.sha256(ckpt.read_bytes()).hexdigest(),'weights':'ema',
        'sf':.18215,'shift':0.,'sde_gamma':0.,'sampler':'heun','steps':a.steps,'cfg':1.,'std_soft_sigma':.65,
        'seeds':seeds,'distinct_characters':len(selected),'selected_eval_indices':selected,'per_image':metadata,
        'limitations':'GT-source proxy is the evaluation reference source, not known generation provenance. GT only supplies evaluation targets; conditions are STD/writer/script/noise.'}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('EXPANDED_GRADIENT_BANK_COMPLETE',len(metadata),flush=True)


if __name__=='__main__':main()
