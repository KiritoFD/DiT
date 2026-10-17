"""CPU geometry + optional CUDA latent audit: no neural training jobs.

Train-split standardized nearest prototypes only. No fitted parameters use test
rows. Results are exploratory, not causal writer/source separation estimates.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import numpy as np
from PIL import Image, ImageFilter
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.analysis.style_fingerprint_features import image_features,latent_features,matched_pairs,VERSION


def extract(task):
    path,variant=task
    with Image.open(path) as im:
        im=im.convert('L').resize((128,128),Image.Resampling.BICUBIC)
        if variant=='blur': im=im.filter(ImageFilter.GaussianBlur(.65))
        return image_features(np.asarray(im)/255.)


def collect(tasks,workers,label):
    values=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i,f in enumerate(pool.map(extract,tasks,chunksize=32)):
            values.append(f)
            if (i+1)%2000==0: print('EXTRACT',label,i+1,'/',len(tasks),flush=True)
    return {key:np.stack([f[key] for f in values]).astype(np.float32) for key in values[0]}


def interval(values,groups,seed=89):
    values=np.asarray(values,float);groups=np.asarray(groups)
    if not len(values):return dict(n=0,accuracy=None,ci95=None,n_groups=0)
    unique,idx=np.unique(groups,return_inverse=True)
    sums=np.bincount(idx,weights=values);counts=np.bincount(idx)
    rng=np.random.default_rng(seed);draw=rng.integers(len(unique),size=(1000,len(unique)))
    bs=sums[draw].sum(1)/counts[draw].sum(1)
    return dict(n=len(values),accuracy=float(values.mean()),ci95=np.quantile(bs,[.025,.975]).tolist(),n_groups=len(unique))


def pair_scores(f,pairs,c,source):
    if not len(pairs):return {'by_char':interval([],[]),'by_source':interval([],[])}
    a,p,n=f[pairs[:,0]],f[pairs[:,1]],f[pairs[:,2]]
    margin=np.mean((a-n)**2,1)-np.mean((a-p)**2,1)
    correct=(margin>0).astype(float)+.5*(margin==0)
    return {'by_char':interval(correct,c[pairs[:,0]]),'by_source':interval(correct,source[pairs[:,0]]),
            'mean_margin':float(margin.mean())}


def evaluate(features,reference,w,s,c,source,train,query,valid,stats):
    tr=train[valid[train]];q=query[valid[query]]
    mu=features[tr].mean(0);sd=np.maximum(features[tr].std(0),.05)
    f=(features-mu)/sd
    slots=sorted(set(zip(w[tr],s[tr])))
    prototypes={slot:f[tr[(w[tr]==slot[0])&(s[tr]==slot[1])]].mean(0) for slot in slots}
    matched,_=matched_pairs(q,tr,w,s,c,source,stats)
    rng=np.random.default_rng(89);unmatched=[];samechar=[];hits=[];hitidx=[]
    for i in q:
        ps=tr[(w[tr]==w[i])&(s[tr]==s[i])&(c[tr]!=c[i])]
        pref=ps[source[ps]!=source[i]];ps=pref if len(pref) else ps
        ns=q[(w[q]!=w[i])&(s[q]==s[i])&(c[q]==c[i])]
        if len(ps) and len(ns):unmatched.append((i,int(rng.choice(ps)),int(rng.choice(ns))))
        allowed=[slot for slot in slots if slot[1]==s[i]]
        if (w[i],s[i]) not in prototypes or len(allowed)<2:continue
        distances=np.array([np.mean((f[i]-prototypes[k])**2) for k in allowed])
        hits.append(allowed[int(distances.argmin())][0]==w[i]);hitidx.append(i)
        if len(ns):
            j=int(rng.choice(ns));proto=prototypes[(w[i],s[i])]
            margin=np.mean((f[j]-proto)**2)-np.mean((f[i]-proto)**2)
            samechar.append((i,float(margin>0)+.5*float(margin==0)))
    arr=np.array(unmatched,dtype=int).reshape(-1,3)
    result={'n_train':len(tr),'n_query':len(q),'dim':features.shape[1],
        'matched':pair_scores(f,matched,c,source),'crosschar':pair_scores(f,arr,c,source),
        'prototype_top1':interval(hits,c[hitidx]),
        'samechar_prototype_2afc':interval([x[1] for x in samechar],c[[x[0] for x in samechar]])}
    # Real/STD AUC using train-only mean residual direction. Script-specific.
    from sklearn.metrics import roc_auc_score
    aucs=[]
    for script in sorted(set(s[q])):
        ti=tr[s[tr]==script];qi=q[s[q]==script]
        if not len(ti) or not len(qi):continue
        direction=((features[ti]-reference[ti])/sd).mean(0)
        a=((features[qi]-mu)/sd)@direction;b=((reference[qi]-mu)/sd)@direction
        aucs.append(float(roc_auc_score(np.r_[np.ones(len(a)),np.zeros(len(b))],np.r_[a,b])))
    result['real_std_auc_script_macro']=float(np.mean(aucs)) if aucs else None
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True);p.add_argument('--workers',type=int,default=24)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--skip-plots',action='store_true',help='Render exported statistics locally if matplotlib is unavailable')
    p.add_argument('--split-root',required=True)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);start=time.time()
    d=np.load('scratch/disc_lat_full.npz',allow_pickle=True)
    w,s,c,source=[d[k].astype(str) for k in ('writer','script','char','work')]
    rows=[r for r in csv.DictReader(open('exp-std/csv/train.csv',encoding='utf-8-sig')) if not (r.get('aug') or '').strip()]
    assert len(rows)==len(w)
    assert all((r['calligrapher'],r['script'],r['character'])==(wi,si,ci) for r,wi,si,ci in zip(rows,w,s,c))
    # Deduplicate exact STD paths; a path collision must not mix script/character.
    stdpaths=sorted({r['std_path'] for r in rows});lookup={x:i for i,x in enumerate(stdpaths)}
    stdidx=np.array([lookup[r['std_path']] for r in rows])
    identities={}
    for r in rows:
        identity=(r['script'],r['character'])
        assert identities.setdefault(r['std_path'],identity)==identity
    cache=out/'features.npz'
    if cache.exists():
        loaded=np.load(cache);assert str(loaded['version'])==VERSION
        real={k[5:]:loaded[k] for k in loaded.files if k.startswith('real_')}
        std={k[4:]:loaded[k] for k in loaded.files if k.startswith('std_')}
        blur={k[5:]:loaded[k] for k in loaded.files if k.startswith('blur_')}
    else:
        real=collect([(r['image_path'],'raw') for r in rows],a.workers,'real')
        std=collect([(x,'raw') for x in stdpaths],a.workers,'std')
        blur=collect([(x,'blur') for x in stdpaths],a.workers,'std_blur')
        z=d['z'];zs=d['z_std'];gi=d['std_idx'].astype(int)
        print('LATENT_STATS',flush=True)
        if a.device=='cuda':
            from tools.analysis.style_fingerprint_gpu import extract as extract_latent
            real['latent']=extract_latent(z)
            ls=extract_latent(zs)
            for raw,features in ((z,real['latent']),(zs,ls)):
                for i in (0,len(raw)//2,len(raw)-1):
                    assert np.allclose(features[i],latent_features(raw[i]),atol=2e-5,rtol=2e-5)
        else:
            real['latent']=np.stack([latent_features(x) for x in z]).astype(np.float32)
            ls=np.stack([latent_features(x) for x in zs]).astype(np.float32)
        # Latent has its own canonical STD key, not image-path indexing.
        real['latent_std']=ls[gi]
        np.savez_compressed(cache,version=VERSION,**{'real_'+k:v for k,v in real.items()},
                            **{'std_'+k:v for k,v in std.items()},**{'blur_'+k:v for k,v in blur.items()})
    families=[k for k in real if k not in ('qc','latent_std')]
    reference={k:(real['latent_std'] if k=='latent' else std[k][stdidx]) for k in families}
    delta={k:real[k]-reference[k] for k in families}
    delta['geometry_combo']=np.concatenate([delta[k] for k in ('quill','hinge','profile')],1)
    reference['geometry_combo']=np.concatenate([reference[k] for k in ('quill','hinge','profile')],1)
    real['geometry_combo']=np.concatenate([real[k] for k in ('quill','hinge','profile')],1)
    delta['all']=np.concatenate([delta[k] for k in families],1)
    reference['all']=np.concatenate([reference[k] for k in families],1)
    real['all']=np.concatenate([real[k] for k in families],1)
    common=np.logical_and.reduce([np.isfinite(real[k]).all(1)&np.isfinite(reference[k]).all(1) for k in real if k in reference])
    split=np.load(Path(a.split_root)/'split_indices.npz')
    works=sorted(set(source));random.Random(0).shuffle(works);held=set(works[:max(1,int(len(works)*.3))])
    protocols={'critic_native_source':(np.flatnonzero(~np.isin(source,list(held))),np.flatnonzero(np.isin(source,list(held)))),
               'character_holdout':(split['train'],split['test_char']),
               'source_holdout':(split['train'],split['test_source'])}
    assert not set(c[split['train']])&set(c[split['test_char']])
    assert not set(zip(w[split['train']],s[split['train']],source[split['train']]))&set(zip(w[split['test_source']],s[split['test_source']],source[split['test_source']]))
    stats=np.load(Path(a.split_root)/'ink_width_stats.npz')['stats']
    report={'version':VERSION,'n_real':len(w),'n_std_paths':len(stdpaths),'n_common':int(common.sum()),
        'data_sha256':hashlib.sha256(Path('scratch/disc_lat_full.npz').read_bytes()).hexdigest(),
        'extractor_sha256':hashlib.sha256(Path(__file__).with_name('style_fingerprint_features.py').read_bytes()).hexdigest(),
        'limitations':['Custom Quill-like and Hinge-like approximations, not paper reproductions.',
            'Source directories are proxies, not verified artworks; writer/source confounding not causally resolved.',
            'Exploratory multiple comparisons; no confirmatory feature winner.',
            'No scattering, learned CNN probes, LDA or mixed-effects fit in this first pass.',
            'Train-only diagonal standardization plus nearest prototypes; histogram dimensions are not equally weighted across families.',
            'Two separate grouped bootstraps (character/source), not a crossed multiway bootstrap.',
            'No original-resolution rendering recovered: raw STD means dataset std_path raster resized to 128.'],
        'coverage':{},'results':{},'std_blur_sensitivity':{},'models':{}}
    for script in sorted(set(s)):
        ix=np.flatnonzero(s==script);chars={char:set(w[ix[c[ix]==char]]) for char in set(c[ix])}
        report['coverage'][script]={'n_chars':len(chars),'shared_chars':sum(len(v)>1 for v in chars.values()),
            'writers':{writer:{'n':int(np.sum(w[ix]==writer)),'sources':len(set(source[ix[w[ix]==writer]])),
                      'shared_char_samples':int(sum(len(chars[c[i]])>1 for i in ix[w[ix]==writer]))} for writer in sorted(set(w[ix]))}}
    for protocol,(tr,q) in protocols.items():
        report['results'][protocol]={}
        for key in delta:
            print('EVALUATE',protocol,key,flush=True)
            # Raw and STD-relative statistics share exactly the same valid rows.
            report['results'][protocol][key]={}
            for mode,f,r in [('raw',real[key],reference[key]),('delta_std',delta[key],np.zeros_like(delta[key]))]:
                report['results'][protocol][key][mode]=evaluate(f,r,w,s,c,source,tr,q,common,stats)
        (out/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    for key in std:
        if key=='qc':continue
        good=np.isfinite(std[key]).all(1)&np.isfinite(blur[key]).all(1)
        report['std_blur_sensitivity'][key]={'n':int(good.sum()),'mean_abs_change':float(np.abs(std[key][good]-blur[key][good]).mean())}
    # Same 187 glyphs, saved generation only. GT paths/pixels must match CSV.
    erows=list(csv.DictReader(open('exp-std/csv/eval200_fixed.csv',encoding='utf-8-sig')))[:187]
    base=Path('assets/runs/v84A_std_cond/20261013-072215-v84A-std-cond')
    dirs={'v84A_200k':base/'eval_samples_ctrl/step0200000/eval200fix',
          'v88_170k':Path('assets/results/v88_writer_glyph_base/20261015-131555-v88-writer-glyph-base/eval_samples_ctrl/step0170000/eval200fix')}
    gt=collect([(r['image_path'],'raw') for r in erows],a.workers,'eval_gt')
    for name,directory in dirs.items():
        for i,r in enumerate(erows):
            with Image.open(directory/f'gt{i}.png') as saved,Image.open(r['image_path']) as orig:
                x=np.asarray(saved.convert('RGB'));y=np.asarray(orig.convert('RGB').resize(saved.size))
                assert np.mean(np.abs(x.astype(float)-y.astype(float)))<1, (name,i,'GT row mismatch')
        generated=collect([(str(directory/f'g{i}.png'),'raw') for i in range(len(erows))],a.workers,name)
        for key in gt:
            if key=='qc':continue
            good=np.isfinite(gt[key]).all(1)&np.isfinite(generated[key]).all(1)
            scale=np.maximum(real[key][common].std(0),.05)
            err=np.mean(np.abs(generated[key][good]-gt[key][good])/scale)
            entry={'n':int(good.sum()),'paired_standardized_abs_error':float(err)}
            if key in ('quill','hinge','width','fft_gray','fft_binary'):
                from scipy.spatial.distance import jensenshannon
                entry['paired_js_distance']=float(np.mean([jensenshannon(x+1e-9,y+1e-9) for x,y in zip(generated[key][good],gt[key][good])]))
            report['models'].setdefault(name,{})[key]=entry
    report['elapsed_seconds']=time.time()-start
    (out/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    if a.skip_plots:
        print('STYLE_FINGERPRINT_STATS_COMPLETE',out,report['elapsed_seconds'],flush=True)
        return
    # Four compact scientific figures. Slot labels are mapped in JSON, not lost.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    slots=sorted(set(zip(w,s)));report['slot_labels']=[list(x) for x in slots]
    selected=['ink','width','quill','hinge','profile','fft_gray','fft_binary','layout','latent','geometry_combo','all']
    tr=split['train'];q=split['test_char']
    matrix=[];residual=[]
    for writer,script in slots:
        ix=np.flatnonzero((w==writer)&(s==script)&common)
        peers=np.flatnonzero((s==script)&common)
        matrix.append([float(((real[k][ix].mean(0)-real[k][peers].mean(0))/np.maximum(real[k][tr[common[tr]]].std(0),.05)).mean()) for k in selected])
        residual.append([float(np.mean(np.abs(delta[k][ix].mean(0))/np.maximum(real[k][tr[common[tr]]].std(0),.05))) for k in selected])
    for title,data,file in [('Descriptive writer offset (content/source confounded)',matrix,'writer_features.png'),('GT - STD mean residual magnitude',residual,'std_residual.png')]:
        fig,ax=plt.subplots(figsize=(12,8));im=ax.imshow(data,aspect='auto',cmap='coolwarm');fig.colorbar(im,ax=ax)
        ax.set_xticks(range(len(selected)),selected,rotation=45,ha='right');ax.set_yticks(range(len(slots)),[f'S{i:02d}' for i in range(len(slots))]);ax.set_title(title)
        fig.tight_layout();fig.savefig(out/file,dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5))
    for protocol in protocols:
        vals=[report['results'][protocol][k]['delta_std']['matched']['by_char']['accuracy'] for k in selected]
        ax.plot(selected,vals,'o-',label=protocol)
    ax.axhline(.5,color='gray',linestyle='--');ax.set_ylim(0,1);ax.tick_params(axis='x',rotation=35);ax.set_title('Matched cross-character 2AFC / STD-relative descriptors');ax.legend();fig.tight_layout();fig.savefig(out/'crosschar_ranking.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5));keys=list(report['models']['v84A_200k'])
    x=np.arange(len(keys))
    for j,(name,metrics) in enumerate(report['models'].items()):ax.bar(x+(j-.5)*.35,[metrics[k]['paired_standardized_abs_error'] for k in keys],.35,label=name)
    ax.set_xticks(x,keys,rotation=35);ax.set_title('Same-glyph GT gap (lower is better; not writer identification)');ax.legend();fig.tight_layout();fig.savefig(out/'model_gaps.png',dpi=150);plt.close(fig)
    (out/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('STYLE_FINGERPRINT_AUDIT_COMPLETE',out,report['elapsed_seconds'],flush=True)


if __name__=='__main__':main()
