"""Same-script cross-character retrieval and grouped uncertainty, no CE-head assumptions."""
import numpy as np


def interval(values,groups,seed=90):
    values=np.asarray(values,dtype=float);groups=np.asarray(groups);keys=np.unique(groups)
    chunks=[values[groups==g] for g in keys];rng=np.random.default_rng(seed)
    if not len(chunks):return {'mean':None,'ci95':None,'n':0,'groups':0}
    boot=[np.concatenate([chunks[j] for j in rng.integers(len(keys),size=len(keys))]).mean() for _ in range(1000)]
    return dict(mean=float(values.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),n=len(values),groups=len(keys))


def retrieval(f,data,split):
    tr=np.asarray(data['splits']['train']);q=np.asarray(data['splits'][split]);labels=data['labels'];chars=data['char'];si=data['si']
    aps=[];hits=[]
    for start in range(0,len(q),128):
        ids=q[start:start+128];scores=f[ids]@f[tr].T
        for j,i in enumerate(ids):
            legal=(si[tr]==si[i])&(chars[tr]!=chars[i]);truth=labels[tr[legal]]==labels[i]
            if not truth.any() or truth.all():continue
            order=truth[np.argsort(-scores[j,legal],kind='stable')]
            aps.append(float(((np.cumsum(order)/(np.arange(len(order))+1))*order).sum()/order.sum()))
            hits.append(bool(order[0]))
    p=np.asarray(data['matched_pairs'][split]);margin=(f[p[:,0]]*f[p[:,1]]).sum(1)-(f[p[:,0]]*f[p[:,2]]).sum(1)
    correct=(margin>0)+.5*(margin==0);used=set();keep=[]
    for i,tri in enumerate(p):
        if not used.intersection(tri):used.update(tri);keep.append(i)
    return dict(crosschar_map=float(np.mean(aps)),crosschar_nearest_top1=float(np.mean(hits)),
        matched_char=interval(correct,data['char'][p[:,0]]),matched_source=interval(correct,data['source'][p[:,0]]),
        image_disjoint=interval(correct[keep],data['char'][p[keep,0]]),mean_margin=float(margin.mean()))


def scores(e,prototypes,allowed,target):
    import torch
    sims=torch.einsum('bd,bkd->bk',e,prototypes).masked_fill(~allowed,-torch.inf)
    target_score=sims.gather(1,target[:,None]).flatten();other=sims.clone();other.scatter_(1,target[:,None],-torch.inf)
    return dict(margin=target_score-other.max(1).values,correct=(sims.argmax(1)==target).float())
