"""Frozen DINOv2 intermediate token codebook + multi-character matched audit.

Allograph-inspired custom pilot, not a reproduction of a named paper.
No neural optimizer; K-means fits training-split tokens only.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.analysis.allograph_stats import make_trials,measure,distance
from tools.analysis.run_style_fingerprint_audit import interval


def single_scores(shape,hist,pairs,chars,sources,kind):
    margins=[]
    for a,p,n in pairs:
        if kind=='combined':
            dp=.5*(distance(shape[a],shape[p],'shape')+distance(hist[a],hist[p],'hist'))
            dn=.5*(distance(shape[a],shape[n],'shape')+distance(hist[a],hist[n],'hist'))
        else:
            x=hist if kind=='hist' else shape
            dp=distance(x[a],x[p],kind);dn=distance(x[a],x[n],kind)
        margins.append(dn-dp)
    margins=np.array(margins);scores=(margins>0).astype(float)+.5*(margins==0)
    return {'by_char':interval(scores,chars[pairs[:,0]]),'by_source':interval(scores,sources[pairs[:,0]])}


def statistics(out,data,feature_sets,control_only=False):
    valid=np.logical_and.reduce([np.isfinite(x).all(1) if not isinstance(x,tuple) else np.isfinite(x[0]).all(1)&np.isfinite(x[1]).all(1) for x,_ in feature_sets.values()])
    filtered=dict(data);filtered['splits']={k:v[valid[v]] for k,v in data['splits'].items()}
    report={'n_common_valid_images':int(valid.sum()),'features':list(feature_sets),'aggregation':{},'single_image_matched':{},
        'limitations':['Custom Allograph-inspired pilot, not a paper reproduction.',
            'Source directories are proxies, not verified artworks; source/writer confounding remains.',
            'Repeated trials reuse GT images and characters; grouped CIs are exploratory.',
            'Writer-pair clusters share writers; source-block bootstrap does not eliminate all dependence.',
            'Whole-word final CLS is a shape proxy, not guaranteed content/style disentanglement.',
            'Pixel D1 is a supervised frozen baseline, unlike pretrained generic DINO.'],
        'no_neural_training':True}
    for protocol,source_holdout in [('character_holdout',False),('source_holdout',True)]:
        trials32,coverage=make_trials(filtered,32,source_holdout=source_holdout)
        cohort={tuple((row['w'],row['v'])) for row in trials32}
        report['aggregation'][protocol]={}
        for k in (1,4,16,32):
            trials,coverage=make_trials(filtered,k,source_holdout=source_holdout)
            common=[t for t in trials if (t['w'],t['v']) in cohort]
            entry={'coverage':coverage,'n_pairs_32_cohort':len(cohort),'n_eligible_pairs':sum(v['eligible'] for v in coverage),
                   'all_eligible':{},'same_32_cohort':{}}
            for key,(features,kind) in feature_sets.items():
                entry['all_eligible'][key]=measure(features,trials,kind)
                entry['same_32_cohort'][key]=measure(features,common,kind)
            report['aggregation'][protocol][str(k)]=entry
        pairs=data['matched_pairs']['test_source' if source_holdout else 'test_char']
        pairs=pairs[valid[pairs].all(1)];report['single_image_matched'][protocol]={}
        for key,(features,kind) in feature_sets.items():
            shape,hist=features if kind=='combined' else (features,features)
            report['single_image_matched'][protocol][key]=single_scores(shape,hist,pairs,data['char'],data['source'],kind)
    path=out/('pixel_control.json' if control_only else 'results.json')
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('ALLOGRAPH_STATISTICS_COMPLETE',path,flush=True)


@torch.no_grad()
def extract_variant(model,gray,variant,batch):
    model.eval();local=[];owners=[];shape=[];n=len(gray)
    mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    for start in range(0,n,batch):
        x=gray[start:start+batch].cuda().float()/255
        raw=F.interpolate(x,size=(224,224),mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
        mask=F.avg_pool2d((raw<.55).float(),14,14).flatten(1)>=.05
        if variant=='binary':x=(x>=.55).float()
        x=F.interpolate(x,size=(224,224),mode='bicubic',align_corners=False,antialias=True).clamp(0,1).repeat(1,3,1,1)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            output=model(pixel_values=(x-mean)/std,output_hidden_states=True)
        tokens=output.hidden_states[6][:,1:]
        assert tokens.shape[1]==256, tokens.shape
        tokens=F.normalize(tokens.float(),dim=-1)
        shape.append(F.normalize(output.last_hidden_state[:,0].float(),dim=-1).cpu().numpy())
        for j in range(len(x)):
            loc=tokens[j][mask[j]].half().cpu().numpy()
            local.append(loc);owners.extend([start+j]*len(loc))
        if start%1024==0:print('DINO_TOKENS',variant,start,'/',n,'allocated_GiB',torch.cuda.max_memory_allocated()/1024**3,flush=True)
    return np.concatenate(local),np.array(owners,np.int32),np.concatenate(shape)


def fit_histograms(tokens,owners,train,n_images):
    from sklearn.cluster import MiniBatchKMeans
    from threadpoolctl import threadpool_limits
    # At most eight deterministic foreground tokens per training image. No
    # heldout token is passed to .fit, normalization or dictionary selection.
    choices=[];starts=np.r_[0,np.flatnonzero(np.diff(owners))+1];ends=np.r_[starts[1:],len(owners)]
    training=set(map(int,train))
    for begin,end in zip(starts,ends):
        if int(owners[begin]) in training:choices.extend(np.linspace(begin,end-1,min(8,end-begin)).astype(int))
    fit=np.asarray(choices,int);assert np.isin(owners[fit],train).all()
    with threadpool_limits(limits=8):
        km=MiniBatchKMeans(n_clusters=128,random_state=89,n_init=3,batch_size=4096,max_iter=100,reassignment_ratio=.01)
        km.fit(tokens[fit].astype(np.float32))
        hist=np.zeros((n_images,128),np.float32)
        for start in range(0,len(tokens),4096):
            assigned=km.predict(tokens[start:start+4096].astype(np.float32));np.add.at(hist,(owners[start:start+4096],assigned),1)
    missing=hist.sum(1)==0;hist/=np.maximum(hist.sum(1,keepdims=True),1);hist[missing]=np.nan
    return hist,km.cluster_centers_,{'n_fit_tokens':len(fit),'n_all_foreground_tokens':len(tokens),'fit_owner_indices':sorted(set(owners[fit].tolist()))}


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--batch',type=int,default=8)
    p.add_argument('--control-only',action='store_true');p.add_argument('--memory-fraction',type=float,default=.12)
    a=p.parse_args();torch.set_num_threads(2);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    base=Path('/root/Workspace/xy/DiT/scratch/writer_pixel_d0d1_20261016_01/assets')
    data=torch.load(base/'data/data.pt',map_location='cpu',weights_only=False)
    pixel=np.load(base/'D1/clean_features.npz')['embedding'];sets={'pixel_D1_global':(pixel,'shape')}
    if a.control_only:statistics(out,data,sets,True);return
    torch.cuda.set_per_process_memory_fraction(a.memory_fraction)
    from transformers import AutoModel
    model=AutoModel.from_pretrained('facebook/dinov2-base',local_files_only=True).eval().requires_grad_(False)
    digest=hashlib.sha256()
    for value in model.state_dict().values():digest.update(value.numpy().tobytes())
    metadata={'model':'facebook/dinov2-base','state_sha256':digest.hexdigest(),'local_block':6,'global_feature':'final CLS',
        'input_size':224,'gray_source':'fixed GT grayscale128, bicubic antialiased224',
        'K':128,'fit_partition':'encoder train only; up to 8 foreground tokens/image',
        'foreground_mask':'original resized gray ink<.55, patch ink fraction>=.05, excludes CLS',
        'batch':a.batch,'per_process_torch_memory_fraction':a.memory_fraction,'variants':{}}
    model=model.cuda()
    for variant in ('gray','binary'):
        cache=out/(variant+'_fingerprints.npz')
        if cache.exists():
            saved=np.load(cache);shape,hist=saved['shape'],saved['hist']
        else:
            tokens,owners,shape=extract_variant(model,data['gray'],variant,a.batch)
            hist,centers,info=fit_histograms(tokens,owners,data['splits']['train'],len(pixel))
            metadata['variants'][variant]=info
            np.savez_compressed(cache,shape=shape,hist=hist,centers=centers,fit_owner_indices=info['fit_owner_indices'])
            del tokens,owners
        sets[variant+'_CLS']=(shape,'shape');sets[variant+'_allograph']=(hist,'hist');sets[variant+'_combined']=((shape,hist),'combined')
    metadata['max_allocated_GiB']=torch.cuda.max_memory_allocated()/1024**3
    (out/'manifest.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    statistics(out,data,sets)


if __name__=='__main__':main()
