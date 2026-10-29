"""Build the frozen E-G reference bank for P3-B style ranking.

Per (writer,script) slot of the Top10 training corpus: sample up to 128 real
images, compute E-G embeddings (256->160 bilinear, ImageNet norm, net.emb,
L2-normalized) and store character labels for per-query exclusion at training
time. Output: assets/p125_20261030/eg_refs.npz  {slot:int -> embs,chars}.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

EG_MEAN=torch.tensor([.485,.456,.406]);EG_STD=torch.tensor([.229,.224,.225])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--csv',default='assets/u100_p1p2_20261025/p2_train.csv')
    p.add_argument('--out',default='assets/p125_20261030/eg_refs.npz')
    p.add_argument('--k',type=int,default=128)
    p.add_argument('--seed',type=int,default=125777)
    a=p.parse_args()
    import sys
    sys.path.insert(0,'.')
    from tools.eval_gen_with_disc import load_evaluator_model
    net,kind,size=load_evaluator_model('assets/style_disc_ce_holdout.pt','cuda')
    if kind!='resnet18':raise ValueError(kind)
    net.eval().requires_grad_(False)
    rows=list(csv.DictReader(open(a.csv,encoding='utf-8-sig')))
    rows=[r for r in rows if not (r.get('aug') or '').strip()]
    rng=np.random.default_rng(a.seed)
    by=defaultdict(list)
    for i,r in enumerate(rows):by[(int(r['calligrapher_id']),int(r['script_id']))].append(i)
    refs={}
    with torch.no_grad():
        for slot,idxs in sorted(by.items()):
            pick=list(rng.choice(len(idxs),min(a.k,len(idxs)),replace=False))
            sel=[rows[idxs[i]] for i in pick]
            batch=torch.from_numpy(np.stack([np.asarray(
                __import__('PIL.Image',fromlist=['Image']).open(r['image_path']).convert('RGB').resize((256,256),__import__('PIL.Image',fromlist=['Image']).Resampling.BICUBIC)).copy()
                for r in sel])).permute(0,3,1,2).float()/255
            out=[]
            for j in range(0,len(batch),64):
                x=F.interpolate(batch[j:j+64].cuda(),(size,size),mode='bilinear',align_corners=False)
                x=(x-EG_MEAN.to(x.device)[None,:,None,None])/EG_STD.to(x.device)[None,:,None,None]
                e=net.emb(x)
                out.append(e/e.norm(dim=-1,keepdim=True).clamp_min(1e-6))
            script_slot={0:0,3:1,4:2}[slot[1]]
            key=f"{slot[0]}_{script_slot}"
            refs[key]=refs.get(key,{'embs':[],'chars':[]})
            refs[key]['embs'].append(torch.cat(out).cpu().numpy())
            for r in sel:refs[key]['chars'].append(r['character'])
    merged={k:{'embs':np.concatenate([np.atleast_2d(x) for x in v['embs']]),
               'chars':v['chars']} for k,v in refs.items()}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    np.savez(a.out,**{f'slot_{k}':v for k,v in merged.items()})
    print('EG bank slots:',len(merged),{k:v['embs'].shape for k,v in list(merged.items())[:4]})


if __name__=='__main__':main()
