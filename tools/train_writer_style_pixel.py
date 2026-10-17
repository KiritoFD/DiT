"""Controlled D0/D1 pixel writer encoders. No generator/STD forward path."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.model.writer_style_pixel import WriterStylePixel,augment_gray
from src.utils.writer_quartets import WriterQuartets,cross_loss
from tools.analysis.style_fingerprint_features import matched_pairs


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(out,splitroot):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise FileExistsError('Preparation target is not empty')
    d=np.load('scratch/disc_lat_full.npz',allow_pickle=True)
    w,s,c,source=[d[k].astype(str) for k in ('writer','script','char','work')]
    rows=[r for r in csv.DictReader(open('exp-std/csv/train.csv',encoding='utf-8-sig')) if not (r.get('aug') or '').strip()]
    assert len(rows)==len(w) and all((r['calligrapher'],r['script'],r['character'])==(ww,ss,cc) for r,ww,ss,cc in zip(rows,w,s,c))
    split=np.load(Path(splitroot)/'split_indices.npz');splits={k:split[k] for k in ('train','val_char','test_char','test_source')}
    allrows=np.concatenate(list(splits.values()));assert len(allrows)==len(set(allrows.tolist()))
    train=splits['train'];assert not set(c[train])&set(c[splits['val_char']])
    assert not set(c[train])&set(c[splits['test_char']])
    assert not set(zip(w[train],s[train],source[train]))&set(zip(w[splits['test_source']],s[splits['test_source']],source[splits['test_source']]))
    slots=sorted(set(zip(w,s)));scripts=sorted(set(s))
    labels=np.array([slots.index(x) for x in zip(w,s)]);si=np.array([scripts.index(x) for x in s])
    sampler=WriterQuartets(train,labels,si,c,source)
    assert not sampler.coverage['slots_without_full_rectangles'],sampler.coverage
    statsfile=Path(splitroot)/'ink_width_stats.npz';stats=np.load(statsfile);digest=sha('scratch/disc_lat_full.npz')
    assert str(stats['data_sha256'])==digest
    def read(row):
        with Image.open(row['image_path']) as im:return np.asarray(im.convert('L').resize((128,128),Image.Resampling.BICUBIC))
    with ThreadPoolExecutor(max_workers=16) as pool:gray=np.stack(list(pool.map(read,rows)))[:,None].copy()
    matched={key:matched_pairs(splits[key],train,w,s,c,source,stats['stats'])[0] for key in ('val_char','test_char','test_source')}
    assert len(matched['test_char'])==847 and len(matched['test_source'])==196
    payload={'gray':torch.from_numpy(gray),'writer':w,'script':s,'char':c,'source':source,'labels':labels,'si':si,
        'slots':slots,'scripts':scripts,'splits':splits,'stats':stats['stats'],'matched_pairs':matched}
    torch.save(payload,out/'data.pt')
    manifest={'data_sha256':digest,'csv_sha256':sha('exp-std/csv/train.csv'),'split_sha256':sha(Path(splitroot)/'split_indices.npz'),
        'stats_sha256':sha(statsfile),'slots':slots,'scripts':scripts,'split_sizes':{k:len(v) for k,v in splits.items()},
        'matched_counts':{k:len(v) for k,v in matched.items()},'quartet_coverage':sampler.coverage,
        'source_limitations':'Source directories are proxies, not verified artworks; nested writer/source effects are not identifiable.',
        'model_inputs':'GT grayscale only. No writer/script/STD inputs. Labels are losses only.',
        'data_cache_sha256':sha(out/'data.pt')}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    np.savez_compressed(out/'split_indices.npz',**splits)
    print('PIXEL_DATA_PREPARED',json.dumps(manifest,ensure_ascii=False),flush=True)


@torch.no_grad()
def embed(model,gray,indices,batch=256,perturb=None,domain='pixel'):
    features=[];pred=[]
    for start in range(0,len(indices),batch):
        ids=indices[start:start+batch];x=gray[torch.as_tensor(ids,device='cuda')].float()
        if domain=='pixel':x=x/255
        if perturb=='blur':x=F.avg_pool2d(F.pad(x,(1,1,1,1),mode='replicate'),3,stride=1)
        if perturb=='ink_gain':x=(1-(1-x)*.85).clamp(0,1) if domain=='pixel' else x*.95
        with torch.autocast('cuda',dtype=torch.bfloat16):o=model(x)
        features.append(o['embedding'].float().cpu().numpy());pred.extend(o['slot_logits'].argmax(1).cpu().tolist())
    return np.concatenate(features),np.array(pred)


def evaluate_features(features,pred,data,query,reference):
    labels,si,chars,source=[data[k] for k in ('labels','si','char','source')]
    nslots=len(data['slots']);proto=[]
    for slot in range(nslots):
        # Source-balanced real training references only.
        ix=reference[labels[reference]==slot];means=[]
        for src in sorted(set(source[ix])):
            e=features[ix[source[ix]==src]].mean(0);means.append(e/max(np.linalg.norm(e),1e-12))
        e=np.mean(means,axis=0);proto.append(e/max(np.linalg.norm(e),1e-12))
    proto=np.array(proto);slot_si=np.array([data['scripts'].index(x[1]) for x in data['slots']])
    hits=[];aps=[];pair_acc=[];rng=np.random.default_rng(89)
    for start in range(0,len(query),128):
        ids=query[start:start+128];scores=features[ids]@features[reference].T
        ps=features[ids]@proto.T
        for j,i in enumerate(ids):
            candidates=(si[reference]==si[i])&(chars[reference]!=chars[i]);relevant=labels[reference[candidates]]==labels[i]
            if relevant.any() and not relevant.all():
                ranked=relevant[np.argsort(-scores[j,candidates],kind='stable')]
                aps.append(float(((np.cumsum(ranked)/(np.arange(len(ranked))+1))*ranked).sum()/ranked.sum()))
            allowed=np.flatnonzero(slot_si==si[i]);hits.append(allowed[ps[j,allowed].argmax()]==labels[i])
            neg=query[(si[query]==si[i])&(chars[query]==chars[i])&(labels[query]!=labels[i])]
            if len(neg):
                k=int(rng.choice(neg));m=float((features[i]-features[k])@proto[labels[i]])
                pair_acc.append(float(m>0)+.5*float(m==0))
    return {'crosschar_map':float(np.mean(aps)),'prototype_top1':float(np.mean(hits)),
            'slot_ce_top1':float(np.mean(pred[query]==labels[query])),
            'prototype_targeted_samechar_2afc':float(np.mean(pair_acc)) if pair_acc else None,
            'note_samechar':'Prototype-targeted, not directly comparable to historical 78.47% logit-based T1.'}


def matched_score(features,pairs):
    m=(features[pairs[:,0]]*features[pairs[:,1]]).sum(1)-(features[pairs[:,0]]*features[pairs[:,2]]).sum(1)
    return float(((m>0).astype(float)+.5*(m==0)).mean())


def train(a):
    out=Path(a.out);out.mkdir(parents=True,exist_ok=bool(a.resume))
    torch.set_num_threads(4);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    torch.backends.cudnn.benchmark=True
    data=torch.load(Path(a.data_root)/'data.pt',map_location='cpu',weights_only=False)
    if a.domain=='latent':
        from src.model.writer_style_latent import build_writer_style_latent,augment_latent,warp_geometry_sdf
        factory,augmentation=lambda n:build_writer_style_latent(n,a.architecture),augment_latent
    else:factory,augmentation=WriterStylePixel,augment_gray
    gray=data['z' if a.domain=='latent' else 'gray'].cuda();tr=data['splits']['train'];val=data['splits']['val_char']
    geometry=a.domain=='latent' and a.architecture=='geometry'
    if geometry:
        sidecar=Path(a.sdf_cache);assert json.loads((sidecar/'manifest.json').read_text())['latent_manifest_sha256']==sha(Path(a.data_root)/'manifest.json')
        sdf=torch.load(sidecar/'unclipped_sdf.pt',map_location='cpu',weights_only=True).cuda()
        assert sdf.shape==(len(gray),1,32,32)
    elif a.geometry_weight:raise ValueError('Geometry loss requires geometry architecture')
    model=factory(len(data['slots'])).cuda();params=sum(p.numel() for p in model.parameters())
    sampler=WriterQuartets(tr,data['labels'],data['si'],data['char'],data['source'],a.seed)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01,fused=True)
    initial_hash=hashlib.sha256()
    for parameter in model.parameters():initial_hash.update(parameter.detach().cpu().numpy().tobytes())
    config={**vars(a),'parameters':params,'data_manifest_sha256':sha(Path(a.data_root)/'manifest.json'),
            'initial_parameter_sha256':initial_hash.hexdigest(),'warmup_ce_epochs':3,'cross_ramp_epochs':10,
            'source_hashes':{str(p):sha(p) for p in [Path(__file__),Path('src/model/writer_style_pixel.py'),Path('src/utils/writer_quartets.py')]},
            'selection':'validation matched cross-character accuracy, tie validation crosschar mAP; tests only after completion',
            'augmentation':('latent affine +-3deg/4pct translation/5pct scale, border padding, gain .95..1.05, noise .01' if a.domain=='latent' else 'affine +-3deg/4pct translation/5pct scale, ink gain .85..1.15, background .03, noise .005, blur sigma.65 p.25')}
    if a.domain=='latent':config['source_hashes']['src/model/writer_style_latent.py']=sha('src/model/writer_style_latent.py')
    if geometry:config['source_hashes']['src/model/writer_style_geometry.py']=sha('src/model/writer_style_geometry.py');config['sdf_manifest_sha256']=sha(sidecar/'manifest.json')
    best=(-1,-1);start=time.time();last=start;times=[];ce_sum=cross_sum=0;beststep=0
    first=1
    if a.resume:
        checkpoint=torch.load(a.resume,map_location='cpu',weights_only=False)
        for key in ('arm','batch','steps','lr','seed','data_manifest_sha256'):
            assert checkpoint['config'][key]==config[key],key
        if geometry:
            for key in ('architecture','geometry_weight','sdf_manifest_sha256'):assert checkpoint['config'][key]==config[key],key
        model.load_state_dict(checkpoint['model'],strict=True);opt.load_state_dict(checkpoint['optimizer'])
        best=tuple(checkpoint['best_validation']);beststep=checkpoint['best_step'];first=checkpoint['step']+1
        torch.set_rng_state(checkpoint['torch_rng']);torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        np.random.set_state(checkpoint['numpy_rng']);random.setstate(checkpoint['python_rng']);sampler.rng.bit_generator.state=checkpoint['sampler_rng']
    else:(out/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    labels=torch.from_numpy(data['labels']).cuda()
    warmup=max(1,math.ceil(3*len(tr)/a.batch));cross_ramp=max(1,math.ceil(10*len(tr)/a.batch))
    def save(name,step):
        torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step,'config':config,
            'best_validation':best,'best_step':beststep,'slots':data['slots'],'scripts':data['scripts'],
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
            'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),'sampler_rng':sampler.rng.bit_generator.state},out/name)
    print('PIXEL_TRAIN_START',a.arm,'parameters',params,'batch',a.batch,'steps',a.steps,flush=True)
    for step in range(first,a.steps+1):
        if a.benchmark:torch.cuda.synchronize();tick=time.perf_counter()
        ids,sampling=sampler.sample(a.batch);ids_gpu=torch.as_tensor(ids,device='cuda')
        raw=gray[ids_gpu].float()
        if geometry:
            x,grid,scale=augmentation(raw,return_transform=True)
            geometry_target=warp_geometry_sdf(sdf[ids_gpu],grid,scale)
        else:x=augmentation(raw if a.domain=='latent' else raw/255)
        model.train();opt.zero_grad(set_to_none=True)
        if step in (1,500,1000):
            trace={'step':step,'sample_ids_sha256':hashlib.sha256(ids.tobytes()).hexdigest(),
                   'augmented_gray_sha256':hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest()}
            if geometry:trace['geometry_target_sha256']=hashlib.sha256(geometry_target.cpu().numpy().tobytes()).hexdigest()
            with (out/'paired_input_trace.jsonl').open('a') as f:f.write(json.dumps(trace)+'\n')
        lr=a.lr*min(step/warmup,1)*(.1+.9*.5*(1+math.cos(math.pi*step/a.steps)))
        for group in opt.param_groups:group['lr']=lr
        ramp=min(max((step-warmup)/cross_ramp,0),1)
        weight=.5*ramp if a.arm!='D0' else 0.
        with torch.autocast('cuda',dtype=torch.bfloat16):o=model(x);ce=F.cross_entropy(o['slot_logits'],labels[ids_gpu])
        cross,lossmetrics=cross_loss(o['embedding'].float(),data['char'][ids])
        style_loss=ce+weight*cross;geometry_loss=style_loss.new_zeros(());geometry_weight=a.geometry_weight*ramp
        gradient_ratio=None
        if geometry:
            geometry_loss=F.l1_loss(o['geometry_sdf'].float(),geometry_target)
            if step%500==0:
                shared=model.stem[0].weight
                gs=torch.autograd.grad(style_loss,shared,retain_graph=True)[0]
                gg=torch.autograd.grad(geometry_loss,shared,retain_graph=True)[0]
                gradient_ratio=float(geometry_weight*gg.norm()/gs.norm().clamp_min(1e-12))
        loss=style_loss+geometry_weight*geometry_loss;loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
        ce_sum+=float(ce.detach());cross_sum+=float(cross.detach())
        if a.benchmark:torch.cuda.synchronize();times.append(time.perf_counter()-tick)
        if step%20==0 or step==a.steps:
            now=time.time();entry={'step':step,'ce':ce_sum/(20 if step%20==0 else step%20),'cross':cross_sum/(20 if step%20==0 else step%20),
                'cross_weight':weight,'lr':lr,'grad_norm':float(grad),'steps_per_second':20/(now-last),
                'elapsed_seconds':now-start,'max_memory_gb':torch.cuda.max_memory_allocated()/1024**3,**sampling,**lossmetrics}
            if geometry:entry.update(geometry_l1=float(geometry_loss.detach()),geometry_weight=geometry_weight,geometry_gradient_ratio_at_stem=gradient_ratio)
            with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(entry)+'\n')
            print('TRAIN',json.dumps(entry),flush=True);last=now;ce_sum=cross_sum=0
        if not a.benchmark and (step%a.eval_every==0 or step==a.steps):
            model.eval();indices=np.r_[tr,val];emb,pr=embed(model,gray,indices,domain=a.domain)
            allfeatures=np.zeros((len(gray),256),np.float32);allpred=np.zeros(len(gray),int)
            allfeatures[indices]=emb;allpred[indices]=pr
            result=evaluate_features(allfeatures,allpred,data,val,tr)
            result['matched_2afc']=matched_score(allfeatures,data['matched_pairs']['val_char']);result['step']=step
            score=(result['matched_2afc'],result['crosschar_map'])
            if score>best:best=score;beststep=step;save('best.pt',step)
            save('last.pt',step)
            with (out/'validation.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
            if step in (1000,3000,5000,10000) or step%2000==0 or step==a.steps:save(f'step{step:07d}.pt',step)
            print('VALIDATION',json.dumps(result),flush=True);last=time.time()
    if a.benchmark:
        median=float(np.median(times[20:]));report={'median_step_seconds':median,'parameters':params,'max_memory_gb':torch.cuda.max_memory_allocated()/1024**3,'batch':a.batch}
        (out/'benchmark.json').write_text(json.dumps(report,indent=2));print('BENCHMARK',json.dumps(report),flush=True)
    print('PIXEL_TRAIN_COMPLETE',a.arm,'best_step',beststep,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true')
    p.add_argument('--data-root',default='assets/data');p.add_argument('--split-root',default='/root/Workspace/xy/DiT/scratch/v89_teacher_v3_20261016_01/assets/encoder')
    p.add_argument('--out',required=True);p.add_argument('--arm',choices=['D0','D1','D2','D3'],default='D1')
    p.add_argument('--batch',type=int,default=512);p.add_argument('--steps',type=int,default=60)
    p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--seed',type=int,default=89)
    p.add_argument('--eval-every',type=int,default=500);p.add_argument('--benchmark',action='store_true')
    p.add_argument('--resume',default='')
    p.add_argument('--domain',choices=['pixel','latent'],default='pixel')
    p.add_argument('--architecture',choices=['d1','geometry'],default='d1')
    p.add_argument('--geometry-weight',type=float,default=0.)
    p.add_argument('--sdf-cache',default='assets/sdf32')
    a=p.parse_args()
    if a.prepare:prepare(a.out,a.split_root)
    else:train(a)
