"""E-G shortcut & nuisance audit (Phase 2b).

Questions answered (attribution, not censorship — width IS legal style):
1. How much writer identity do low-level statistics alone carry?
   -> character-grouped logistic probe on the 20-dim nuisance vector.
2. Does E-G verification survive statistical control of those statistics?
   -> residualize E-G on nuisance (train folds), rerun K64 retrieval.
3. Do content-preserving perturbations move E-G like a shortcut would?
   -> small translate/scale, binarize-vs-gray, width-normalization controls.

All learned parts split train/valid by character fold. Outputs
nuisance_ablation.csv + nuisance_audit.json under docs/1030/eval_style_v2/.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from .common import load_real_rows,save_json,read_json,stable_fold
from . import features_hq as hq
from . import metrics as M

SALT='esv2-nuisance-v1'
N_FOLDS=5


def load_feats(machine='48',out='docs/1030/eval_style_v2'):
    root=Path(f'assets/eval_style_v2_cache/{machine}')
    q_eg=np.load(root/'real_queries_eg.npz')['E-G']
    q_nuis=np.load(Path(out)/f'nuisance_queries_{machine}.npy')
    return q_eg,q_nuis


def bank_arrays(machine='48',out='docs/1030/eval_style_v2'):
    from . import features_eg as eg
    rows,feats=eg.bank_eg(machine)
    bank_nuis=np.load(Path(out)/f'nuisance_bank_{machine}.npy')
    return rows,feats,bank_nuis


def residualize(emb,nuis,train_mask):
    """Remove the linearly explainable part of emb given nuisance, fold-wise."""
    x=np.concatenate([nuis,np.ones((len(nuis),1),dtype=np.float32)],axis=1)
    out=emb.copy().astype(np.float32)
    coef,_,_,_=np.linalg.lstsq(x[train_mask],emb[train_mask],rcond=None)
    out=emb-x@coef
    return out.astype(np.float32)


def probe_auc(nuis,labels,train_mask,valid_mask,seed=3):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    mu=nuis[train_mask].mean(0);sd=nuis[train_mask].std(0)+1e-6
    x=(nuis-mu)/sd
    clf=LogisticRegression(max_iter=2000,C=1.0,random_state=seed)
    clf.fit(x[train_mask],labels[train_mask])
    proba=clf.predict_proba(x[valid_mask])
    classes=clf.classes_
    present=[w for w in np.unique(labels[valid_mask]) if w in classes]
    idx={w:i for i,w in enumerate(classes)}
    y_true=np.array([1 if labels[i]==w else 0 for i in np.nonzero(valid_mask)[0] for w in present])
    y_score=np.concatenate([proba[:,idx[w]] for w in present])
    return float(roc_auc_score(y_true,y_score))


def perturb(gray,kind):
    import numpy as np
    from scipy import ndimage as ndi
    if kind=='shift_scale':
        out=np.zeros_like(gray)
        dy,dx,scale=6,5,1.04
        h,w=gray.shape
        ys=(np.arange(h)-h/2)/scale+h/2-dy;xs=(np.arange(w)-w/2)/scale+w/2-dx
        ys=np.clip(ys,0,h-1);xs=np.clip(xs,0,w-1)
        out=gray[np.ix_(ys.astype(int),xs.astype(int))]
        return out.astype(np.float32)
    if kind=='binarize':
        mask,_=hq.ink_parts(gray)
        return np.where(mask,0.,1.).astype(np.float32)
    if kind=='width_norm':
        mask,_=hq.ink_parts(gray)
        width=2*ndi.distance_transform_edt(mask)
        med=float(np.median(width[mask&(width>0)])) if mask.any() else 1.
        factor=8./max(med,1.)
        er=int(max(0,round((med-8)/2))) if factor<1 else 0
        out=gray
        if er>0:
            m2=ndi.binary_erosion(mask,np.ones((3,3),bool),iterations=er)
            out=np.where(m2,0.,1.).astype(np.float32)
        return out
    raise ValueError(kind)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--machine',default='48')
    p.add_argument('--out',default='docs/1030/eval_style_v2')
    a=p.parse_args()
    out=Path(a.out)
    q_eg,q_nuis=load_feats(a.machine)
    bank_rows,bank_eg,bank_nuis=bank_arrays(a.machine)
    from .verify import real_queries,bank_hq,concat_hq
    queries=real_queries(bank_rows)
    assert len(queries)==len(q_eg),(len(queries),len(q_eg))
    rows=[];audit={}
    # 1) low-level probe, character-grouped folds
    labels=np.array([int(r['calligrapher_id']) for r in queries])
    chars=np.array([r['character'] for r in queries])
    fold=np.array([stable_fold(c,SALT,N_FOLDS) for c in chars])
    fold_aucs=[]
    for f in range(N_FOLDS):
        valid=fold==f;train=~valid
        if valid.sum()<10:continue
        fold_aucs.append(probe_auc(q_nuis,labels,train,valid))
    probe_macro=float(np.mean(fold_aucs))
    rows.append(dict(control='low_level_probe_writer_auc',value=round(100*probe_macro,2),
        note=f'character-grouped {N_FOLDS}-fold logistic on 20-dim nuisance; chance=50'))
    audit['low_level_probe_auc']=round(100*probe_macro,2)
    audit['low_level_probe_folds']=[round(100*v,2) for v in fold_aucs]
    # 2) residualized E-G retrieval (fold-wise residual on queries; bank residualized with its own fit)
    x=np.concatenate([q_nuis,np.ones((len(q_nuis),1),dtype=np.float32)],axis=1)
    resid_q=np.zeros_like(q_eg)
    for f in range(N_FOLDS):
        m=fold==f;train=~m
        if train.sum()<10:train=np.ones(len(q_nuis),dtype=bool)
        coef,_,_,_=np.linalg.lstsq(x[train],q_eg[train],rcond=None)
        resid_q[m]=q_eg[m]-x[m]@coef
    xb=np.concatenate([bank_nuis,np.ones((len(bank_nuis),1),dtype=np.float32)],axis=1)
    coef,_,_,_=np.linalg.lstsq(xb,bank_eg,rcond=None)
    resid_bank=(bank_eg-xb@coef).astype(np.float32)
    for name,qf in (('E-G_raw',q_eg),('E-G_residualized',resid_q)):
        bankf=bank_eg if name=='E-G_raw' else resid_bank
        res,_=M.evaluate_feature(qf,queries,bankf,bank_rows,64)
        rows.append(dict(control=f'{name}_retrieval_K64',value=res.get('auc'),
            n=res.get('n_valid'),note='same protocol as real_encoder_validation.json'))
        audit[name]=res
    # 3) perturbation controls on a fixed real subset (bank rows are cheap to reload)
    subset=bank_rows[:300]
    base=[];pert={k:[] for k in ('shift_scale','binarize','width_norm')}
    for r in subset:
        g=hq.load_gray(r['image_path'])
        f0=hq.extract(g)
        base.append(np.concatenate([f0['hinge_s'],f0['hinge_l'],f0['quill']]))
        for k in pert:
            fp=hq.extract(perturb(g,k))
            pert[k].append(np.concatenate([fp['hinge_s'],fp['hinge_l'],fp['quill']]))
    base=np.asarray(base)
    def drift(mtx):
        d=np.linalg.norm(mtx-base,axis=1)
        ref=np.linalg.norm(base-np.roll(base,1,axis=0),axis=1)
        return float(np.median(d)/max(np.median(ref),1e-9))
    for k,mtx in pert.items():
        mtx=np.asarray(mtx)
        rows.append(dict(control=f'hq_drift_{k}',value=round(drift(mtx),3),
            note='median HQ feature distance to original / median neighbor distance; ~0 means invariant'))
    save_json(out/'nuisance_audit.json',audit)
    with open(out/'nuisance_ablation.csv','w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['control','value','n','note'])
        w.writeheader()
        for r in rows:r.setdefault('n','');w.writerow(r)
    print(json.dumps(audit,ensure_ascii=False))


if __name__=='__main__':main()
