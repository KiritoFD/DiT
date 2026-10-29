"""Writer verification, style-distribution and PRDC metrics on frozen features."""
import numpy as np
from sklearn.metrics import roc_auc_score,roc_curve,average_precision_score
from tools.eval_unpaired.panels.clean import fid_low_rank


def roc(labels,scores):
    fpr,tpr,threshold=roc_curve(labels,scores,drop_intermediate=False)
    at=np.argmin(np.abs(fpr-(1-tpr)))
    return dict(auc=float(roc_auc_score(labels,scores)),eer=float((fpr[at]+1-tpr[at])/2),
        tar_at_eval_far01=float(tpr[fpr<=.01].max()),eer_threshold=float(threshold[at]))


def verification(x,bank,rows,refs,k=64):
    decisions=[];labels=[];scores=[]
    for feature,row in zip(x,rows):
        groups={}
        for i,r in enumerate(refs):
            if r['script']==row['script'] and r['character']!=row['character']:
                groups.setdefault(r['calligrapher'],[]).append(i)
        groups={key:ids[:k] for key,ids in groups.items() if len(ids)>=k}
        if row['calligrapher'] not in groups or len(groups)<2:continue
        ids=[i for indices in groups.values() for i in indices]
        yy=np.array([refs[i]['calligrapher']==row['calligrapher'] for i in ids]);ss=bank[ids]@feature
        stats=roc(yy,ss);stats.update(writer=row['calligrapher'],character=row['character'],
            script=row['script'],ap=float(average_precision_score(yy,ss)),positive=int(yy.sum()),negative=int((~yy).sum()))
        decisions.append(stats);labels.extend(yy);scores.extend(ss)
    if not decisions:return dict(n_queries=0)
    writers=sorted({d['writer'] for d in decisions})
    macro={metric:float(np.mean([np.mean([d[metric] for d in decisions if d['writer']==w]) for w in writers]))
           for metric in ['auc','eer','ap','tar_at_eval_far01']}
    return dict(n_queries=len(decisions),macro_writer=macro,pooled=roc(labels,scores),rows=decisions,
                interpretation='EER/TAR from evaluation ROC, not a calibrated deployment threshold; pair scores are correlated')


def kid(a,b):
    if min(len(a),len(b))<2:return None
    a=a.astype(np.float64);b=b.astype(np.float64);d=a.shape[1]
    aa=(1+a@a.T/d)**3;bb=(1+b@b.T/d)**3;ab=(1+a@b.T/d)**3
    return float((aa.sum()-aa.trace())/(len(a)*(len(a)-1))+
                 (bb.sum()-bb.trace())/(len(b)*(len(b)-1))-2*ab.mean())


def density_coverage(real,gen,k=5):
    if len(real)<=k or not len(gen):return None
    rr=np.maximum(0,2-2*real@real.T);np.fill_diagonal(rr,np.inf)
    radii=np.partition(rr,k-1,axis=1)[:,k-1];rg=np.maximum(0,2-2*real@gen.T)
    return dict(density=float((rg<radii[:,None]).sum(0).mean()/k),
                coverage=float(np.mean(rg.min(1)<radii)),k=k,n_real=len(real),n_gen=len(gen))


def distributions(x,bank,rows,refs):
    fake=[];real=[];real_b=[];slots={}
    for slot in sorted({(r['calligrapher'],r['script']) for r in rows}):
        ii=[i for i,r in enumerate(rows) if (r['calligrapher'],r['script'])==slot]
        chars={rows[i]['character'] for i in ii}
        jj=[j for j,r in enumerate(refs) if (r['calligrapher'],r['script'])==slot and r['character'] not in chars]
        n=min(len(ii),len(jj)//2)
        if not n:continue
        a=x[ii[:n]];b=bank[jj[:n]];c=bank[jj[n:2*n]]
        slots['|'.join(slot)]=dict(n=n,centroid_distance=float(np.linalg.norm(a.mean(0)-b.mean(0))),
            real_split_centroid=float(np.linalg.norm(c.mean(0)-b.mean(0))))
        fake.append(a);real.append(b);real_b.append(c)
    if not fake:return dict(n=0)
    a,b,c=map(np.concatenate,[fake,real,real_b]);by={}
    for key,v in slots.items():by.setdefault(key.split('|')[0],[]).append(v)
    return dict(n=len(a),style_fid=fid_low_rank(a,b),style_kid=kid(a,b),
        real_split_style_fid=fid_low_rank(c,b),real_split_style_kid=kid(c,b),
        cn_centroid_macro=float(np.mean([np.mean([r['centroid_distance'] for r in rs]) for rs in by.values()])),
        density_coverage=density_coverage(b,a),real_split_density_coverage=density_coverage(b,c),slots=slots,
        reference='training-reference, disjoint characters; equal writer-script counts and N; unit-L2 feature FID/KID; exploratory small N')
