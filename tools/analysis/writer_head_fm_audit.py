"""Heldout GT, target-aware2x2 writer swap. No parameter updates or CFG."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.analysis.v90_p0 import read_rows,legal_wrong_slots,sha
from src.eval.model_io import load_model_from_ckpt
from src.eval.in_mem_eval import _get_cache
from src.eval.sampler import model_kwargs_for_slice
from src.utils.vae_scaling import resolve_vae_scaling


def swapped(model,xt,time,good,bad):
    captures=[]
    def capture(module,args,kwargs,output):captures.append((args,dict(kwargs)))
    hook=model.final_layer.register_forward_hook(capture,with_kwargs=True)
    try:
        original_a=model(xt,time,**good)
        original_b=model(xt,time,**bad)
    finally:hook.remove()
    (a,ka),(b,kb)=captures
    def head(hidden,rest,kw):
        return model.unpatchify(model.final_layer(hidden,*rest,**kw))[:,:4].float()
    values={'AA':head(a[0],a[1:],ka),'AB':head(a[0],b[1:],kb),
            'BA':head(b[0],a[1:],ka),'BB':head(b[0],b[1:],kb)}
    assert torch.allclose(values['AA'],original_a[:,:4].float(),atol=1e-5,rtol=1e-5)
    assert torch.allclose(values['BB'],original_b[:,:4].float(),atol=1e-5,rtol=1e-5)
    return values


def metrics(values,target):
    losses={k:(v-target).square().flatten(1).mean(1) for k,v in values.items()}
    b=values['BA']-values['AA'];h=values['AB']-values['AA']
    cross=values['BB']-values['BA']-values['AB']+values['AA']
    total=values['BB']-values['AA']
    norm=values['AA'].flatten(1).norm(dim=1).clamp_min(1e-8)
    result={f'L_{k}':v for k,v in losses.items()}
    result.update(delta_head=losses['AB']-losses['AA'],delta_backbone=losses['BA']-losses['AA'],
                  delta_total=losses['BB']-losses['AA'],
                  loss_interaction=losses['BB']-losses['BA']-losses['AB']+losses['AA'],
                  cosine=torch.nn.functional.cosine_similarity(b.flatten(1),h.flatten(1)))
    for key,value in [('backbone',b),('head',h),('interaction',cross),('total',total)]:
        result['mag_'+key]=value.flatten(1).norm(dim=1)/norm
    return {k:v.cpu().numpy() for k,v in result.items()}


def summarize(records):
    result={k:float(np.mean([r[k] for r in records])) for k in records[0] if k.startswith(('L_','delta_','mag_')) or k in ('cosine','loss_interaction')}
    groups=defaultdict(list)
    for row in records:groups[row['character']].append(row)
    keys=list(groups);rng=np.random.default_rng(90)
    # Cluster bootstrap, retaining all repeated seeds and times per character.
    weights=np.array([len(groups[k]) for k in keys])
    draws=rng.integers(0,len(keys),(2000,len(keys)))
    for field in ('delta_head','delta_backbone','delta_total'):
        sums=np.array([sum(r[field] for r in groups[k]) for k in keys])
        bootstrap=sums[draws].sum(1)/weights[draws].sum(1)
        result[field+'_ci95']=np.quantile(bootstrap,[.025,.975]).tolist()
        result[field+'_positive_fraction']=float(np.mean([r[field]>0 for r in records]))
    result.update(n_records=len(records),n_characters=len(keys))
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',action='append',required=True,help='label=checkpoint')
    parser.add_argument('--out',required=True)
    parser.add_argument('--seeds',type=int,default=3)
    parser.add_argument('--batch',type=int,default=16)
    parser.add_argument('--memory-fraction',type=float,default=1.)
    args=parser.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    if args.memory_fraction<1:torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    train=read_rows('exp-std/csv/train_top10_aug_sym.csv')
    rows=read_rows('exp-std/csv/eval200_fixed.csv')
    overlap={}
    for field in ('image_path','src_image_path'):
        shared={r[field] for r in rows if r.get(field)} & {r[field] for r in train if r.get(field)}
        if shared:raise ValueError(f'Train/eval overlap: {field}: {sorted(shared)[:5]}')
        overlap[field]=0
    reps,options=legal_wrong_slots(rows,train);rep_keys=sorted(reps)
    combined=rows+[reps[k] for k in rep_keys]
    condition_path=out/'conditions.csv'
    with condition_path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in combined))))
        writer.writeheader();writer.writerows(combined)
    checkpoints=dict(x.split('=',1) for x in args.checkpoint)
    # Resolve the actual VAE scale contract from the first checkpoint.
    first=torch.load(next(iter(checkpoints.values())),map_location='cpu',weights_only=False)
    sf,shift=resolve_vae_scaling(first['args']);del first
    from diffusers import AutoencoderKL
    from PIL import Image
    vae=AutoencoderKL.from_pretrained('data/pretrained/sd-vae-ft-ema').cuda().eval().requires_grad_(False)
    generator=torch.Generator(device='cuda').manual_seed(9001)
    zz=[]
    for start in range(0,len(rows),args.batch):
        images=[]
        for row in rows[start:start+args.batch]:
            image=Image.open(row['image_path']).convert('RGB').resize((256,256),Image.Resampling.BICUBIC)
            images.append(torch.from_numpy(np.array(image,copy=True)).permute(2,0,1))
        x=torch.stack(images).cuda().float()/127.5-1
        posterior=vae.encode(x).latent_dist
        zz.append(((posterior.sample(generator=generator)-shift)*sf).cpu())
    z=torch.cat(zz);torch.save(z,out/'gt_latents.pt')
    del vae,x,posterior
    report=dict(n_gt=len(rows),overlap=overlap,seeds=args.seeds,times=[.25,.4,.6,.8],
                convention='xt=(1-t)*GT+t*noise; target=noise-GT; network_t=1000*t',
                posterior='fixed FP32 VAE posterior sample, seed9001',sf=sf,shift=shift,
                gt_latents_sha256=sha(out/'gt_latents.pt'),models={},
                limitations=['Teacher-forced FM diagnostic, not generation quality.',
                             'Image/source-image paths held out; artwork/source domains are not held out.',
                             'Character-cluster bootstrap; seeds/times are repeated measures.'])
    for label,path in checkpoints.items():
        model,model_args=load_model_from_ckpt(path,device='cuda',use_ema=True)
        model.eval().requires_grad_(False)
        assert resolve_vae_scaling(model_args)==(sf,shift)
        model_args.std_soft_root='exp-std/data/std_soft_glyphs_v1'
        model_args.std_soft_eval_sigma=.65
        cache=_get_cache(str(condition_path),len(combined),None,'',model_args)
        def kwargs(indices,glyphs):
            return model_kwargs_for_slice(model,
                yc=torch.tensor([cache['conds'][i][0] for i in indices],device='cuda'),
                yh=torch.tensor([cache['conds'][i][1] for i in glyphs],device='cuda'),
                hier_slice=tuple([[axis[i] for i in indices] for axis in cache['hier_conds']]),
                skel_slice=cache['skels_latent'][glyphs],device='cuda')
        records=[]
        for seed in range(args.seeds):
            noise=torch.randn(z.shape,generator=torch.Generator().manual_seed(90+seed))
            for t in report['times']:
                for start in range(0,len(rows),args.batch):
                    indices=list(range(start,min(start+args.batch,len(rows))))
                    wrong=[options[i][(i+seed)%len(options[i])] for i in indices]
                    other=[len(rows)+rep_keys.index(k) for k in wrong]
                    good,bad=kwargs(indices,indices),kwargs(other,indices)
                    for key in set(good)-{'y_callig','y_callig_raw','y_pair'}:
                        assert torch.equal(good[key],bad[key]),key
                    assert torch.all(good['y_callig']!=bad['y_callig'])
                    gt=z[indices].cuda();eps=noise[indices].cuda();xt=(1-t)*gt+t*eps
                    values=swapped(model,xt,torch.full((len(indices),),t*1000,device='cuda'),good,bad)
                    measured=metrics(values,eps-gt)
                    for j,i in enumerate(indices):
                        records.append(dict(row=i,character=rows[i]['character'],writer=rows[i]['calligrapher'],
                            source=rows[i].get('source','unknown'),script=rows[i]['script'],
                            wrong_writer=wrong[j][0],seed=seed,t=t,**{k:float(v[j]) for k,v in measured.items()}))
                print(label,seed,t,'complete',flush=True)
        (out/(label+'_records.json')).write_text(json.dumps(records,ensure_ascii=False))
        report['models'][label]=dict(checkpoint=path,sha256=sha(path),weights='ema',
            all=summarize(records),by_time={str(t):summarize([r for r in records if r['t']==t]) for t in report['times']},
            by_source={s:summarize([r for r in records if r['source']==s]) for s in sorted({r['source'] for r in records})})
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        del model,cache,values,good,bad
    print(json.dumps({k:v['by_time'] for k,v in report['models'].items()},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
