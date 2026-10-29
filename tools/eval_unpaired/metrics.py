"""Non-paired statistics. Negative unbiased MMD/KID estimates are retained."""
import numpy as np


def mmd(a,b,bandwidth=None):
    if min(len(a),len(b))<2:return None
    z=np.concatenate([a,b]);d=np.maximum(0,2-2*z@z.T)
    if bandwidth is None:
        upper=d[np.triu_indices(len(z),1)];positive=upper[upper>1e-10]
        if not len(positive):return 0.
        bandwidth=float(np.median(positive))
    k=np.mean([np.exp(-d/(bandwidth*s)) for s in (.5,1.,2.)],axis=0)
    n=len(a);aa=k[:n,:n];bb=k[n:,n:];ab=k[:n,n:]
    return float((aa.sum()-np.trace(aa))/(n*(n-1))+
                 (bb.sum()-np.trace(bb))/(len(b)*(len(b)-1))-2*ab.mean())


def retrieval(query,features,rows,refs,k=64,exclude_source=False):
    """Same-script competition; exclude same Unicode and exact image/source."""
    decisions=[]
    for x,row in zip(query,rows):
        by={}
        for j,r in enumerate(refs):
            if r['script']!=row['script'] or r['character']==row['character']:continue
            if r.get('pixel_sha256') and r.get('pixel_sha256')==row.get('pixel_sha256'):continue
            if exclude_source and (not row.get('source') or r.get('source')==row['source']):continue
            by.setdefault(r['calligrapher'],[]).append(j)
        scores={};counts={}
        for writer,ids in by.items():
            # Stable content-independent bank selection, never nearest-K selection.
            ids=ids[:k];counts[writer]=len(ids)
            if len(ids)<k:continue
            similarity=features[ids]@x
            scores[writer]=float(np.sort(similarity)[-min(5,len(ids)):].mean())
        target=row['calligrapher'];others=[v for w,v in scores.items() if w!=target]
        if target not in scores or not others:
            decisions.append(dict(valid=False,counts=counts));continue
        margin=scores[target]-max(others)
        decisions.append(dict(valid=True,writer=target,character=row['character'],script=row['script'],
            top1=float(margin>0),margin=margin,afc=float(np.mean([scores[target]>v for v in others])),
            competitors=len(others),counts=counts))
    valid=[d for d in decisions if d['valid']]
    if not valid:return dict(n=0,top1=None,margin=None,afc=None,rows=decisions)
    writers=sorted({d['writer'] for d in valid})
    return dict(n=len(valid),top1=float(np.mean([d['top1'] for d in valid])),
        macro_top1=float(np.mean([np.mean([d['top1'] for d in valid if d['writer']==w]) for w in writers])),
        margin=float(np.mean([d['margin'] for d in valid])),afc=float(np.mean([d['afc'] for d in valid])),
        intervals={key:grouped_ci([d[key] for d in valid],
            [dict(calligrapher=d['writer'],character=d['character']) for d in valid]) for key in ['top1','margin','afc']},rows=decisions)


def distribution(query,features,rows,refs):
    result={}
    for slot in sorted({(r['calligrapher'],r['script']) for r in rows}):
        qi=[i for i,r in enumerate(rows) if (r['calligrapher'],r['script'])==slot]
        chars={rows[i]['character'] for i in qi}
        ri=[i for i,r in enumerate(refs) if (r['calligrapher'],r['script'])==slot and r['character'] not in chars]
        if min(len(qi),len(ri))<3:continue
        a=query[qi];b=features[ri[:128]]
        # Bandwidth depends on real data only, shared for correct/wrong-writer tests.
        distances=np.maximum(0,2-2*b@b.T);v=distances[np.triu_indices(len(b),1)];v=v[v>1e-10]
        bandwidth=float(np.median(v)) if len(v) else 1.
        mid=len(b)//2
        cross=np.maximum(0,2-2*a@b.T)
        within=np.maximum(0,2-2*b@b.T);np.fill_diagonal(within,np.inf)
        radii=np.partition(within,min(3,len(b)-1)-1,axis=1)[:,min(3,len(b)-1)-1]
        wrong=[]
        for other in sorted({r['calligrapher'] for r in refs if r['script']==slot[1] and r['calligrapher']!=slot[0]}):
            ids=[i for i,r in enumerate(refs) if r['script']==slot[1] and r['calligrapher']==other and r['character'] not in chars]
            if len(ids)>=3:wrong.append(mmd(a,features[ids[:128]],bandwidth))
        result['|'.join(slot)]=dict(n_gen=len(a),n_real=len(b),mmd2=mmd(a,b,bandwidth),
            real_split_mmd2=mmd(b[:mid],b[mid:],bandwidth),bandwidth=bandwidth,
            wrong_writer_mmd2=float(np.mean(wrong)) if wrong else None,
            coverage=float(np.mean(cross.min(0)<=radii)),
            precision=float(np.mean(np.any(cross<=radii[None,:],axis=1))))
    writers={}
    for key,value in result.items():writers.setdefault(key.split('|')[0],[]).append(value)
    by_writer={key:{metric:float(np.mean([v[metric] for v in values]))
                   for metric in ['mmd2','coverage','precision']} for key,values in writers.items()}
    values=[r['mmd2'] for r in by_writer.values()]
    return dict(macro_mmd2=float(np.mean(values)) if values else None,by_writer=by_writer,
        slot_macro_mmd2=float(np.mean([r['mmd2'] for r in result.values()])) if result else None,slots=result,
        reference='generator-training reference; excludes queried characters; not heldout-distribution proof')


def grouped_ci(values,rows,seed=90):
    groups={}
    for v,r in zip(values,rows):groups.setdefault((r['calligrapher'],r['character']),[]).append(v)
    v=np.array([np.mean(v) for v in groups.values()]);rng=np.random.default_rng(seed)
    if not len(v):return None
    means=np.array([rng.choice(v,len(v),replace=True).mean() for _ in range(1000)])
    return dict(mean=float(np.mean(values)),lo=float(np.quantile(means,.025)),hi=float(np.quantile(means,.975)),
                groups=len(v),method='writer-character cluster bootstrap; screening only')
