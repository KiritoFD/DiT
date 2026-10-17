"""One final local-statistics round. Frozen protocols; never launches training."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.analysis.local_stroke_features import extract_local,triplet_margin,VERSION,FAMILIES
from tools.analysis.style_fingerprint_features import matched_pairs
from tools.analysis.run_style_fingerprint_audit import interval


def worker(path):
    with Image.open(path) as im:
        return extract_local(np.asarray(im.convert('L').resize((128,128),Image.Resampling.BICUBIC))/255.)


def summarize(margin,pairs,c,source):
    outcomes=(margin>0).astype(float)+.5*(margin==0)
    return {'by_char':interval(outcomes,c[pairs[:,0]]),'by_source':interval(outcomes,source[pairs[:,0]]),
            'mean_margin':float(margin.mean()) if len(margin) else None}


def disjoint(pairs):
    """Deterministic image-disjoint subset; do not call enlarged repeats independent."""
    used=set();keep=[]
    for i,triplet in enumerate(pairs):
        if len(set(triplet))==3 and not used.intersection(triplet):
            used.update(triplet);keep.append(i)
    return np.array(keep,int)


def normalize_cuda(values,train):
    import torch
    x=torch.as_tensor(values,device='cuda',dtype=torch.float32)
    tr=x[torch.as_tensor(train,device='cuda')];finite=torch.isfinite(tr)
    count=finite.sum(0).clamp_min(1)
    mu=torch.nan_to_num(tr,nan=0.).sum(0)/count
    var=torch.where(finite,(tr-mu)**2,0.).sum(0)/count
    return (x/var.sqrt().clamp_min(.03)).cpu().numpy()


def diagnostic_global(old,tr,q,pairs,common_keep,w,s,c,source,stats):
    results={};rng=np.random.default_rng(89);changed=[];original=[]
    for i,j,k in pairs:
        choices=q[(w[q]!=w[i])&(s[q]==s[i])&(c[q]!=c[i])]
        dd=np.abs(stats[choices]-stats[i]);choices=choices[(dd[:,0]<=.02)&(dd[:,1]<=.15*max(stats[i,1],1e-6))]
        if not len(choices):continue
        changed.append([i,j,int(rng.choice(choices))]);original.append([i,j,k])
    changed=np.array(changed,int).reshape(-1,3);original=np.array(original,int).reshape(-1,3)
    for key in ('quill','hinge'):
        x=old['real_'+key]
        methods={'old_std_mse':x/np.maximum(x[tr].std(0),.05),
                 'small_floor_std_mse':x/np.maximum(x[tr].std(0),.001),
                 'hellinger_mse':np.sqrt(np.maximum(x,0))}
        results[key]={}
        for name,f in methods.items():
            def score(pp):
                aa,po,ne=f[pp[:,0]],f[pp[:,1]],f[pp[:,2]]
                return np.mean((aa-ne)**2,1)-np.mean((aa-po)**2,1)
            margin=score(pairs)
            results[key][name]={'fixed':summarize(margin,pairs,c,source),
                'same_local_valid_subset':summarize(margin[common_keep],pairs[common_keep],c,source),
                'content_control_samechar_negative':summarize(score(original),original,c,source),
                'content_control_diffchar_negative':summarize(score(changed),changed,c,source),
                'reverse_distance_accuracy_NOT_valid_style_score':float((margin<0).mean()+.5*(margin==0).mean())}
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    p.add_argument('--workers',type=int,default=24);a=p.parse_args();start=time.time()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    root=Path('assets/style_fingerprint_20261016');old=np.load(root/'features.npz')
    d=np.load('scratch/disc_lat_full.npz',allow_pickle=True)
    w,s,c,source=[d[k].astype(str) for k in ('writer','script','char','work')]
    rows=[r for r in csv.DictReader(open('exp-std/csv/train.csv',encoding='utf-8-sig')) if not (r.get('aug') or '').strip()]
    assert len(rows)==len(w) and all((r['calligrapher'],r['script'],r['character'])==(ww,ss,cc) for r,ww,ss,cc in zip(rows,w,s,c))
    digest=hashlib.sha256(Path('scratch/disc_lat_full.npz').read_bytes()).hexdigest()
    assert digest==json.loads((root/'results.json').read_text())['data_sha256']
    cache=out/'local_features.npz'
    if not cache.exists():
        values=[]
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            for i,x in enumerate(pool.map(worker,[r['image_path'] for r in rows],chunksize=16)):
                values.append(x)
                if (i+1)%2000==0:print('LOCAL_PATCHES',i+1,'/',len(rows),flush=True)
        np.savez_compressed(cache,version=VERSION,data_sha256=digest,
            **{k:np.stack([v[k] for v in values]) for k in (*FAMILIES,'counts')})
    data=np.load(cache);assert str(data['version'])==VERSION and str(data['data_sha256'])==digest
    splitroot=Path('scratch/v89_teacher_v3_20261016_01/assets/encoder')
    split=np.load(splitroot/'split_indices.npz');cached=np.load(splitroot/'ink_width_stats.npz')
    assert str(cached['data_sha256'])==digest;stats=cached['stats']
    original_good=np.logical_and.reduce([np.isfinite(old[k]).all(1) for k in old.files if k.startswith('real_') and k not in ('real_qc',)])
    protocols={'character_holdout':split['test_char'],'source_holdout':split['test_source'],
               'expanded_character_holdout':np.r_[split['val_char'],split['test_char']]}
    train=split['train'][original_good[split['train']]]
    assert not set(c[train])&set(c[protocols['expanded_character_holdout']])
    groupvalid=np.logical_and.reduce([np.isfinite(data[k]).all(-1) for k in FAMILIES])
    normalized={k:normalize_cuda(data[k],train) for k in FAMILIES}
    # Equal feature-family weight in the combined distance, not dimension count.
    normalized['combined']=np.concatenate([normalized[k]/np.sqrt(FAMILIES[k]) for k in FAMILIES],-1)
    report={'version':VERSION,'n_images':len(w),'data_sha256':digest,
        'extractor_sha256':hashlib.sha256(Path(__file__).with_name('local_stroke_features.py').read_bytes()).hexdigest(),
        'predeclared':{'min_shared_groups':2,'min_patches_per_group':3,'max_patches_per_group':32,
            'normalization':'train-only per-group standard deviation, floor=.03; missing groups masked, not zeros',
            'pass':'combined character >=.60; source >.50 with both grouped CI lower >.50; n>=100/50',
            'middle':'combined character >=.55; inspect complementary families, no automatic fit/train',
            'fail':'stop hand-crafted statistics; no automatic representation/critic training'},
        'limitations':['Source directories are not verified artworks.',
            'Grouped bootstrap separately by query character and query source; references are reused.',
            'Image-disjoint subsets remove exact sample reuse but cannot establish independent artworks.',
            'Expanded set adds previously unused validation characters, not repeated triplets or new independent sources.',
            'Custom tangent/curvature approximation excludes junctions and low-anisotropy points.',
            'No reversed-distance score is used for gating.'],
        'protocols':{}}
    stored={}
    for protocol,query in protocols.items():
        q=query[original_good[query]];pairs,cross=matched_pairs(q,train,w,s,c,source,stats)
        if protocol=='character_holdout':assert len(pairs)==847,len(pairs)
        if protocol=='source_holdout':assert len(pairs)==196,len(pairs)
        assert np.all(w[pairs[:,0]]==w[pairs[:,1]]) and np.all(c[pairs[:,0]]!=c[pairs[:,1]])
        assert np.all(s[pairs[:,0]]==s[pairs[:,1]]) and np.all(s[pairs[:,0]]==s[pairs[:,2]])
        assert np.all(w[pairs[:,0]]!=w[pairs[:,2]]) and np.all(c[pairs[:,0]]==c[pairs[:,2]])
        shared=groupvalid[pairs].all(1).sum(1);keep=shared>=2;pp=pairs[keep]
        independent=disjoint(pp)
        entry={'nominal_n':len(pairs),'common_valid_n':int(keep.sum()),'source_cross_positive_n':cross,
            'common_shared_groups_quantiles':np.quantile(shared[keep],[0,.5,1]).tolist() if keep.any() else [],
            'image_disjoint_n':len(independent),'families':{},
            'global_diagnostic':diagnostic_global(old,train,q,pairs,keep,w,s,c,source,stats)}
        entry['sampling_checks']={
            'max_positive_ink_diff':float(np.abs(stats[pairs[:,0],0]-stats[pairs[:,1],0]).max()),
            'max_negative_ink_diff':float(np.abs(stats[pairs[:,0],0]-stats[pairs[:,2],0]).max()),
            'max_positive_relative_width_diff':float((np.abs(stats[pairs[:,0],1]-stats[pairs[:,1],1])/stats[pairs[:,0],1]).max()),
            'max_negative_relative_width_diff':float((np.abs(stats[pairs[:,0],1]-stats[pairs[:,2],1])/stats[pairs[:,0],1]).max()),
            'positive_references_all_train':bool(np.isin(pairs[:,1],train).all()),
            'common_query_characters':len(set(c[pp[:,0]])),
            'common_query_sources':len(set(source[pp[:,0]]))}
        for key,f in normalized.items():
            margin,count=triplet_margin(f,pp,groupvalid)
            entry['families'][key]={'main':summarize(margin,pp,c,source),
                'image_disjoint':summarize(margin[independent],pp[independent],c,source)}
            stored[protocol+'_'+key+'_margins']=margin
        stored[protocol+'_nominal_pairs']=pairs;stored[protocol+'_common_keep']=keep
        stored[protocol+'_pairs']=pp;stored[protocol+'_image_disjoint_indices']=independent
        report['protocols'][protocol]=entry
        print('LOCAL_RESULT',protocol,json.dumps(entry['families']),flush=True)
    ch=report['protocols']['character_holdout'];so=report['protocols']['source_holdout']
    def acc(entry):return entry['families']['combined']['main']['by_char']['accuracy'] or 0
    source_lower=min(so['families']['combined']['main'][key]['ci95'][0] for key in ('by_char','by_source'))
    char_lower=min(ch['families']['combined']['main'][key]['ci95'][0] for key in ('by_char','by_source'))
    if acc(ch)>=.60 and acc(so)>.50 and source_lower>.50 and char_lower>.50 and ch['common_valid_n']>=100 and so['common_valid_n']>=50:
        decision='PASS_FOR_CONSIDERATION_ONLY_NO_TRAINING_STARTED'
    elif acc(ch)>=.55:decision='BORDERLINE_REVIEW_COMPLEMENTARITY_NO_TRAINING'
    else:decision='FAIL_STOP_HANDCRAFTED_STATISTICS_NO_TRAINING'
    report['decision']=decision;report['elapsed_seconds']=time.time()-start
    np.savez_compressed(out/'triplets_and_margins.npz',**stored)
    (out/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('LOCAL_STROKE_AUDIT_COMPLETE',decision,report['elapsed_seconds'],flush=True)


if __name__=='__main__':main()
