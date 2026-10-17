"""Ink/width-matched cross-character, cross-source teacher evaluation.

Evaluation GT is never exported as a training bank. Selection uses only pixel
statistics and metadata, NOT embedding scores. Positive references come only
from the encoder training split and must be a different character.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.model.style_encoder import build_style_encoder


def ink_width(path):
    from PIL import Image
    from scipy.ndimage import distance_transform_edt
    from skimage.morphology import skeletonize
    gray=np.asarray(Image.open(path).convert('L').resize((128,128),Image.Resampling.BICUBIC))/255.
    ink=gray<.55; skeleton=skeletonize(ink)
    if skeleton.sum()<8:
        return float(ink.mean()),float('nan')
    return float(ink.mean()),float(np.median(2*distance_transform_edt(ink)[skeleton]))


def matched_pairs(query,train,w,s,c,source,stats,ink_limit=.02,width_relative=.15):
    pairs=[];cross_source=0
    for i in query:
        if not np.isfinite(stats[i]).all():
            continue
        def filter_match(indices):
            delta=np.abs(stats[indices]-stats[i])
            ok=np.isfinite(delta).all(1)&(delta[:,0]<=ink_limit)&(delta[:,1]<=width_relative*max(stats[i,1],1e-6))
            valid=indices[ok]
            return valid
        p=train[(w[train]==w[i])&(s[train]==s[i])&(c[train]!=c[i])]
        p=filter_match(p)
        preferred=p[source[p]!=source[i]]
        p=preferred if len(preferred) else p
        n=query[(w[query]!=w[i])&(s[query]==s[i])&(c[query]==c[i])]
        n=filter_match(n)
        if not len(p) or not len(n):
            continue
        def nearest(indices):
            distance=(stats[indices,0]-stats[i,0])**2/ink_limit**2+(stats[indices,1]-stats[i,1])**2/max(width_relative*stats[i,1],1e-6)**2
            return int(indices[np.argmin(distance)])
        j,k=nearest(p),nearest(n)
        pairs.append((int(i),j,k));cross_source+=source[j]!=source[i]
    return np.asarray(pairs,dtype=np.int64).reshape(-1,3),cross_source


def score_pairs(features,pairs):
    if not len(pairs):
        return {'n':0,'accuracy':None,'ci95':None}
    a,p,n=features[pairs[:,0]],features[pairs[:,1]],features[pairs[:,2]]
    margin=(a*p).sum(1)-(a*n).sum(1)
    outcomes=(margin>0).astype(float)+.5*(margin==0)
    accuracy=float(outcomes.mean());count=len(outcomes)
    # Wilson interval: descriptive pair-level uncertainty, pairs not independent.
    denom=1+1.96**2/count
    center=(accuracy+1.96**2/(2*count))/denom
    radius=1.96*np.sqrt(accuracy*(1-accuracy)/count+1.96**2/(4*count**2))/denom
    return {'n':count,'accuracy':accuracy,'ci95':[float(center-radius),float(center+radius)],'mean_margin':float(margin.mean())}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--teacher-run',required=True)
    p.add_argument('--baseline-run',default='')
    p.add_argument('--model-name',default='teacher')
    p.add_argument('--data',default='scratch/disc_lat_full.npz')
    a=p.parse_args();torch.set_num_threads(4)
    run=Path(a.teacher_run);d=np.load(a.data,allow_pickle=True)
    w,s,c,source=[d[k].astype(str) for k in ('writer','script','char','work')]
    split=np.load(run/'split_indices.npz')
    train=split['train'];queries={key:split[key] for key in ('test_char','test_source')}
    indices=np.arange(len(w))
    rows=[r for r in csv.DictReader(open('exp-std/csv/train.csv',encoding='utf-8-sig')) if not (r.get('aug') or '').strip()]
    assert len(rows)==len(w)
    for i in indices:
        assert (rows[i]['calligrapher'].strip(),rows[i]['script'].strip(),rows[i]['character'].strip())==(w[i],s[i],c[i])
    cache=run/'ink_width_stats.npz'
    if cache.exists():
        cached=np.load(cache)
        manifest=json.loads((run/'manifest.json').read_text())
        assert str(cached['data_sha256'])==manifest['data_sha256']
        stats=cached['stats']
        assert stats.shape==(len(w),2)
    else:
        stats=np.full((len(w),2),np.nan)
        with ThreadPoolExecutor(max_workers=6) as executor:
            for i,value in zip(indices,executor.map(ink_width,[rows[i]['image_path'] for i in indices])):
                stats[i]=value
        manifest=json.loads((run/'manifest.json').read_text())
        np.savez_compressed(cache,stats=stats,data_sha256=manifest['data_sha256'])
        print('INK_WIDTH_CACHE_COMPLETE',cache,flush=True)
    pairs={};counts={}
    for key,query in queries.items():
        pairs[key],counts[key]=matched_pairs(query,train,w,s,c,source,stats)
        print('MATCHED_PAIRS',key,len(pairs[key]),'of',len(query),flush=True)
    z=torch.from_numpy(d['z'].copy());g=1-torch.from_numpy(d['g_std'].copy()).float()[:,None]/255
    gi=d['std_idx'].astype(np.int64)
    report={'ink_max_abs_diff':.02,'skeleton_width_max_relative_diff':.15,'image_size':128,
            'positive_bank':'encoder train split only; different character; different source preferred',
            'negative':'same character, same script, different writer, from held-out queries only',
            'limitations':['source directories are not verified artwork identities','Wilson CI is pair-level; reused references induce dependence'],
            'models':{}}
    for name,folder in [(a.model_name,run)]+([('baseline',Path(a.baseline_run))] if a.baseline_run else []):
        ck=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
        model=build_style_encoder(len(ck['slots']),len(ck['scripts']),ck.get('architecture','legacy')).cuda().eval()
        model.load_state_dict(ck['model'],strict=True)
        features=np.zeros((len(w),256),np.float32)
        si=np.array([ck['scripts'].index(str(value)) for value in s])
        with torch.no_grad():
            for start in range(0,len(indices),256):
                ii=indices[start:start+256]
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    output=model(z[ii].cuda().float(),g[gi[ii]].cuda(),torch.from_numpy(si[ii]).cuda())
                features[ii]=output['embedding'].cpu().numpy()
        metrics={key:{**score_pairs(features,value),'n_query':len(queries[key]),
                      'positive_cross_source_n':int(counts[key])} for key,value in pairs.items()}
        report['models'][name]=metrics
        del model;torch.cuda.empty_cache()
        print('MATCHED_RESULT',name,json.dumps(metrics),flush=True)
    (run/'matched_teacher.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print('MATCHED_TEACHER_COMPLETE',flush=True)


if __name__=='__main__':
    main()
