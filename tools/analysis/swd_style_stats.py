"""Finite, content-matched sliced W2 audit; no fitted neural networks."""
import numpy as np
import torch


def overlapping_membership_bootstrap(within,between,correct,memberships,seed=89):
    """Poisson cluster weights shared by every occurrence of a character/source.

    A trial spanning several clusters receives their average weight. This is
    a sensitivity interval, not a claim that overlapping trials are independent.
    """
    keys=sorted({x for row in memberships for x in row});index={x:i for i,x in enumerate(keys)}
    rng=np.random.default_rng(seed);cluster=rng.poisson(1,size=(1000,len(keys)))
    weights=np.stack([cluster[:,[index[x] for x in sorted(set(row))]].mean(1) for row in memberships],1)
    valid=weights.sum(1)>0;weights=weights[valid]
    ratio=(weights@between)/np.maximum(weights@within,1e-12)
    acc=(weights@correct)/weights.sum(1)
    return {'n_clusters':len(keys),'R_ci95':np.quantile(ratio,[.025,.975]).tolist(),
            'two_afc_ci95':np.quantile(acc,[.025,.975]).tolist(),
            'note':'Shared Poisson weights across repeated character/source memberships; exploratory overlap-aware sensitivity interval.'}


def projection_directions(dim, count=128, seed=89):
    rng=np.random.default_rng(seed)
    v=rng.standard_normal((dim,count)).astype(np.float32)
    return v/np.maximum(np.linalg.norm(v,axis=0,keepdims=True),1e-12)


def train_normalization(tokens, train):
    # Each image has exactly the same number of equally weighted samples.
    sample=np.asarray(tokens[train],dtype=np.float64)
    mean=sample.mean(axis=(0,1));std=sample.std(axis=(0,1)).clip(1e-3)
    return mean.astype(np.float32),std.astype(np.float32)


def mixture_quantiles(projected, ids, device='cuda'):
    """Exact empirical mixture, NOT the mean of per-image quantile functions.

    projected: image x region x projection x samples_per_region. All images
    contribute equal mass. Region distributions remain separate.
    """
    x=torch.as_tensor(np.asarray(projected[ids]),device=device,dtype=torch.float32)
    # ids = batch x number_of_characters
    b,k,r,p,s=x.shape
    return x.permute(0,2,3,1,4).reshape(b,r,p,k*s).sort(-1).values


def swd_from_quantiles(a,b):
    assert a.shape==b.shape
    return (a-b).square().mean(dim=(-1,-2,-3))


def summarize_distances(within,between,trials,seed=89):
    within=np.asarray(within);between=np.asarray(between)
    if not len(trials):return {'n_trials':0,'R':None,'two_afc':None}
    outcome=(between>within).astype(float)+.5*(between==within)
    # shape = trial x anchor (two writer directions per trial)
    w=within.mean(1);b=between.mean(1);correct=outcome.mean(1)
    def boot(keys):
        keys,idx=np.unique(keys,return_inverse=True);n=len(keys)
        counts=np.bincount(idx);ww=np.bincount(idx,weights=w)
        bb=np.bincount(idx,weights=b);aa=np.bincount(idx,weights=correct)
        draw=np.random.default_rng(seed).integers(n,size=(1000,n))
        ratios=bb[draw].sum(1)/np.maximum(ww[draw].sum(1),1e-12)
        acc=aa[draw].sum(1)/counts[draw].sum(1)
        return {'n_clusters':n,'R_ci95':np.quantile(ratios,[.025,.975]).tolist(),
                'two_afc_ci95':np.quantile(acc,[.025,.975]).tolist()}
    pairs=[f'{t["w"]}:{t["v"]}' for t in trials]
    sources=[t['source_block'] for t in trials]
    characters=['|'.join(t['test_chars']) for t in trials]
    # B-side writer identities are exchanged as entire character bundles,
    # coherently across all trials of a writer pair. NOT a global permutation.
    _,idx=np.unique(pairs,return_inverse=True);rng=np.random.default_rng(seed)
    swaps=rng.integers(0,2,size=(2000,idx.max()+1))[:,idx].astype(bool)
    null=np.where(swaps,1-correct,correct).mean(1)
    leave_one_out=[]
    for i in range(idx.max()+1):
        keep=idx!=i
        if keep.any():leave_one_out.append(float(b[keep].mean()/max(w[keep].mean(),1e-12)))
    return {'n_trials':len(trials),'n_writer_pairs':len(set(pairs)),
            'R':float(b.mean()/max(w.mean(),1e-12)),'within':float(w.mean()),'between':float(b.mean()),
            'two_afc':float(correct.mean()),'pair_bootstrap':boot(pairs),
            'source_bootstrap':boot(sources),'character_set_bootstrap':boot(characters),
            'overlapping_character_bootstrap':overlapping_membership_bootstrap(w,b,correct,[t['test_chars'] for t in trials]),
            'overlapping_source_bootstrap':overlapping_membership_bootstrap(w,b,correct,[t['source_block'].split('|') for t in trials]),
            'per_anchor_between_within_ratio_quantiles':np.quantile(between/np.maximum(within,1e-12),[.1,.5,.9]).tolist(),
            'leave_one_pair_out_R_range':[min(leave_one_out),max(leave_one_out)] if leave_one_out else [],
            'pair_bundle_label_swap':{'draws':2000,'p_ge_observed':float((1+(null>=correct.mean()-1e-12).sum())/2001),
                                     'null_ci95':np.quantile(null,[.025,.975]).tolist(),
                                     'note':'Conditional pair-block label swap, not globally consistent writer relabeling. Cannot remove source confounding.'}}


def global_permutation_diagnostic(distances,trials,slot_scripts,seed=89,draws=500):
    """Permute B-side writer slots globally within script; coverage is explicit.

    distances[t,anchor_index,target_slot]. Missing target character coverage is
    NaN. Compare each draw ONLY to observed outcomes on its identical valid
    trials. Coverage-varying results are diagnostics, not exact permutation p.
    """
    if not len(trials):return {'n_trials':0}
    w=np.array([t['w'] for t in trials]);v=np.array([t['v'] for t in trials]);ar=np.arange(len(w))
    own=np.stack((distances[ar,0,w],distances[ar,1,v]),1)
    other=np.stack((distances[ar,0,v],distances[ar,1,w]),1)
    obs=((other>own)+.5*(other==own)).mean(1)
    rng=np.random.default_rng(seed);records=[]
    for _ in range(draws):
        perm=np.arange(len(slot_scripts))
        for script in sorted(set(slot_scripts)):
            ids=np.where(np.asarray(slot_scripts)==script)[0];perm[ids]=rng.permutation(ids)
        po=np.stack((distances[ar,0,perm[w]],distances[ar,1,perm[v]]),1)
        ne=np.stack((distances[ar,0,perm[v]],distances[ar,1,perm[w]]),1)
        valid=np.isfinite(po).all(1)&np.isfinite(ne).all(1)
        if valid.any():
            acc=((ne[valid]>po[valid])+.5*(ne[valid]==po[valid])).mean()
            records.append((float(valid.mean()),float(acc),float(obs[valid].mean())))
    good=[r for r in records if r[0]>=.8]
    return {'draws':draws,'draws_with_any_coverage':len(records),'draws_with_ge80pct_coverage':len(good),
            'coverage_quantiles':np.quantile([r[0] for r in records],[0,.5,1]).tolist() if records else [],
            'mean_observed_minus_permuted_on_same_valid_trials':float(np.mean([r[2]-r[1] for r in good])) if good else None,
            'note':'Global within-script B-side writer permutation; absent shared-character bundles are not imputed. Variable-coverage diagnostic, NOT an exact p-value or a source-confounding test.'}
