"""Writer verification metrics for eval_style_v2.

Reproduces the eval_unpaired retrieval convention exactly (same-script
competition, query-character and pixel-identity exclusion, stable bank order,
top5-mean per candidate writer), then adds pooled AUC/EER and per-query AP.
No composite score; paired deltas are handled in paired.py.
"""
import numpy as np


def candidate_scores(x,row,feats,ref_rows,k):
    by={}
    for j,r in enumerate(ref_rows):
        if r['script']!=row['script'] or r['character']==row['character']:continue
        if r.get('pixel_sha256') and row.get('pixel_sha256') and r['pixel_sha256']==row['pixel_sha256']:continue
        by.setdefault(r['calligrapher'],[]).append(j)
    scores={};counts={}
    for writer,ids in by.items():
        ids=ids[:k];counts[writer]=len(ids)
        if len(ids)<k:continue
        sim=feats[ids]@x
        scores[writer]=float(np.sort(sim)[-min(5,len(ids)):].mean())
    return scores,counts


def collect_decisions(query_feats,query_rows,feats,ref_rows,k,writer_labels=None):
    """Per-query candidate scores; writer_labels overrides targets for permutation."""
    out=[];invalid=0
    for i,(x,row) in enumerate(zip(query_feats,query_rows)):
        target=(writer_labels.get(i,writer_labels.get(row.get('img_id'))) if writer_labels else row['calligrapher'])
        scores,counts=candidate_scores(x,row,feats,ref_rows,k)
        if target not in scores or not [w for w in scores if w!=target]:
            invalid+=1;continue
        out.append(dict(target=target,row=row,idx=i,scores=scores))
    return out


def decisions_to_valid(decisions):
    valid=[]
    for d in decisions:
        target=d['target'];others=[v for w,v in d['scores'].items() if w!=target]
        if target not in d['scores'] or not others:continue
        margin=d['scores'][target]-max(others)
        ranking=sorted(d['scores'],key=lambda w:-d['scores'][w])
        valid.append(dict(margin=margin,target=target,rank=ranking.index(target)+1,
            character=d['row']['character'],writer=d['row']['calligrapher'],
            script=d['row']['script'],scores=d['scores'],row=d['row']))
    return valid


def auc_eer_from_decisions(decisions):
    """Pooled target-vs-competitor AUC and EER over all valid queries."""
    pos=[];neg=[]
    for d in decisions:
        target=d['target']
        pos.append(d['scores'][target])
        neg+= [v for w,v in d['scores'].items() if w!=target]
    if not pos or not neg:return None,None
    pos=np.asarray(pos);neg=np.asarray(neg)
    diff=pos[:,None]-neg[None,:]
    auc=float((diff>0).mean()+0.5*(diff==0).mean())
    # ROC/EER on pooled samples
    pooled=np.concatenate([pos,neg]);labels=np.concatenate([np.ones(len(pos)),np.zeros(len(neg))])
    order=np.argsort(-pooled,kind='stable');labels=labels[order]
    tp=np.cumsum(labels);fp=np.cumsum(1-labels)
    tpr=tp/max(tp[-1],1);fpr=fp/max(fp[-1],1)
    fnr=1-tpr
    idx=np.argmin(np.abs(fpr-fnr))
    eer=float((fpr[idx]+fnr[idx])/2)
    return auc,eer


def map_macro(valid):
    if not valid:return None
    per_q=[1.0/v['rank'] for v in valid]
    writers=sorted({v['writer'] for v in valid})
    macro=float(np.mean([np.mean([1.0/v['rank'] for v in valid if v['writer']==w]) for w in writers]))
    return dict(macro=macro,pooled=float(np.mean(per_q)))


def char_cluster_bootstrap_auc(valid,n=200,seed=7):
    """Cluster bootstrap over (writer,character); returns 2.5/97.5 percentiles of AUC."""
    rng=np.random.default_rng(seed)
    clusters={}
    for v in valid:
        clusters.setdefault((v['writer'],v['character']),[]).append(v)
    keys=list(clusters)
    if len(keys)<8:return None
    stats=[]
    for _ in range(n):
        pick=rng.choice(len(keys),len(keys),replace=True)
        sample=[v for i in pick for v in clusters[keys[i]]]
        auc,_=auc_eer_from_decisions(sample)
        if auc is not None:stats.append(auc)
    if len(stats)<n//2:return None
    return dict(lo=float(np.percentile(stats,2.5)),hi=float(np.percentile(stats,97.5)))


def evaluate_feature(query_feats,query_rows,feats,ref_rows,k,permutation_seed=None):
    """Full metric block for one feature/K; permutation control via label shuffle."""
    import numpy as np
    labels=None
    if permutation_seed is not None:
        rng=np.random.default_rng(permutation_seed)
        ids=[r['img_id'] for r in query_rows]
        targets=[r['calligrapher'] for r in query_rows]
        rng.shuffle(targets)
        labels=dict(zip(ids,targets))
    decisions=collect_decisions(query_feats,query_rows,feats,ref_rows,k,writer_labels=labels)
    valid=decisions_to_valid(decisions)
    auc,eer=auc_eer_from_decisions(decisions)
    mp=map_macro(valid)
    result=dict(n_valid=len(valid),k=k)
    if auc is not None:
        result.update(auc=round(100*auc,2),eer=round(100*eer,2))
        if mp:result.update(map_macro=round(100*mp['macro'],2))
    if permutation_seed is None:
        ci=char_cluster_bootstrap_auc(valid)
        if ci:result.update(auc_ci95=[round(100*ci['lo'],2),round(100*ci['hi'],2)])
    return result,decisions
