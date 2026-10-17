"""Matched-character multi-image style fingerprints; exploratory, no network fit."""
from itertools import combinations
import numpy as np


def l2(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)


def make_trials(data,k,source_holdout=False,repeats=30,seed=89,restrict_pairs=None):
    """Same A/B character sets across writers; A and B disjoint, no repeats.

    A is always training-only. B is a character or source heldout. Sources are
    directory proxies, never claimed to be verified artworks.
    """
    labels,scripts,chars,sources=[data[x] for x in ('labels','si','char','source')]
    aidx=data['splits']['train'];bidx=data['splits']['test_source' if source_holdout else 'test_char']
    rng=np.random.default_rng(seed);rows=[];coverage=[]
    for w,v in combinations(sorted(set(labels[aidx])),2):
        if data['slots'][w][1]!=data['slots'][v][1]:continue
        if restrict_pairs is not None and (w,v) not in restrict_pairs:continue
        aa={slot:{c:ids[chars[ids]==c] for c in set(chars[ids])} for slot in (w,v) for ids in [aidx[labels[aidx]==slot]]}
        bb={slot:{c:ids[chars[ids]==c] for c in set(chars[ids])} for slot in (w,v) for ids in [bidx[labels[bidx]==slot]]}
        ca=sorted(set(aa[w])&set(aa[v]));cb=sorted(set(bb[w])&set(bb[v]))
        eligible=len(cb)>=k and len(set(ca)-set(cb))>=k if not source_holdout else len(cb)>=k and len(ca)>=k and len(set(ca)|set(cb))>=2*k
        coverage.append({'w':int(w),'v':int(v),'shared_A_chars':len(ca),'shared_B_chars':len(cb),'eligible':bool(eligible)})
        if not eligible:continue
        for _ in range(repeats):
            # Rejection uses metadata only, not any feature outcomes.
            for attempt in range(100):
                cbatch=rng.choice(cb,k,replace=False);available=sorted(set(ca)-set(cbatch))
                if len(available)>=k:break
            else:continue
            abatch=rng.choice(available,k,replace=False)
            ids=[]
            for bank,slot,cs in ((aa,w,abatch),(bb,w,cbatch),(aa,v,abatch),(bb,v,cbatch)):
                ids.append([int(rng.choice(bank[slot][str(c)])) for c in cs])
            aw,bw,av,bv=map(np.asarray,ids)
            if source_holdout:
                assert not set(sources[aw])&set(sources[bw])
                assert not set(sources[av])&set(sources[bv])
            assert set(chars[aw])==set(chars[av]) and set(chars[bw])==set(chars[bv])
            assert not set(chars[aw])&set(chars[bw])
            rows.append({'w':int(w),'v':int(v),'ids':ids,
                'source_block':'|'.join(sorted(set(sources[bw])|set(sources[bv])))})
    return rows,coverage


def distance(a,b,kind):
    if kind=='shape':return float(np.clip(1-l2(a)@l2(b),0,2)/2)
    if kind=='hist':
        a=np.maximum(a,0);b=np.maximum(b,0);a=a/max(a.sum(),1e-12);b=b/max(b.sum(),1e-12);m=(a+b)/2
        return float(.5*(np.sum(a*np.log((a+1e-12)/(m+1e-12)))+np.sum(b*np.log((b+1e-12)/(m+1e-12))))/np.log(2))
    raise ValueError(kind)


def measure(features,trials,kind):
    within=[];between=[];correct=[];pairs=[];sources=[]
    for t in trials:
        if kind=='combined':
            shape,hist=features
            groups=[(shape[i].mean(0),hist[i].mean(0)) for i in t['ids']]
            def d(a,b):return .5*(distance(a[0],b[0],'shape')+distance(a[1],b[1],'hist'))
        else:
            groups=[features[i].mean(0) for i in t['ids']]
            def d(a,b):return distance(a,b,kind)
        aw,bw,av,bv=groups
        ww=np.array([d(aw,bw),d(av,bv)]);bb=np.array([d(aw,bv),d(av,bw)])
        within.append(ww.mean());between.append(bb.mean());correct.append(float(np.mean((bb>ww)+.5*(bb==ww))))
        pairs.append(f'{t["w"]}:{t["v"]}');sources.append(t['source_block'])
    if not within:return {'n_trials':0,'R':None,'two_afc':None,'note':'Insufficient shared-character coverage; no replacement sampling.'}
    within=np.array(within);between=np.array(between);correct=np.array(correct)
    def bootstrap(keys):
        keys,idx=np.unique(keys,return_inverse=True);counts=np.bincount(idx)
        sw=np.bincount(idx,weights=within);sb=np.bincount(idx,weights=between);sa=np.bincount(idx,weights=correct)
        draw=np.random.default_rng(89).integers(len(keys),size=(1000,len(keys)))
        ratios=sb[draw].sum(1)/np.maximum(sw[draw].sum(1),1e-12)
        acc=sa[draw].sum(1)/counts[draw].sum(1)
        return {'n_clusters':len(keys),'R_ci95':np.quantile(ratios,[.025,.975]).tolist(),'two_afc_ci95':np.quantile(acc,[.025,.975]).tolist()}
    return {'n_trials':len(trials),'n_writer_pairs':len(set(pairs)),
        'R':float(between.mean()/max(within.mean(),1e-12)),'within':float(within.mean()),'between':float(between.mean()),
        'two_afc':float(correct.mean()),'pair_cluster_bootstrap':bootstrap(pairs),'source_block_bootstrap':bootstrap(sources),
        'limitations':'Trials reuse images/characters. Writer-pair clusters share writers; CIs exploratory, not independent-source proof.'}
