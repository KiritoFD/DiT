"""Metadata-only larger image-disjoint SOURCE subset; preserve original 196 too.

Select maximum coverage over 200 seeded greedy trials, never using features,
predictions or correctness. Apply the SAME chosen triplets to Pixel and Latent.
"""
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def main():
    d=torch.load('assets/data/data.pt',map_location='cpu',weights_only=False)
    w,s,c,src,stats=[d[k] for k in ('labels','si','char','source','stats')]
    tr=d['splits']['train'];q=d['splits']['test_source'];pools={}
    for i in q:
        def match(ids):
            delta=np.abs(stats[ids]-stats[i]);return ids[np.isfinite(delta).all(1)&(delta[:,0]<=.02)&(delta[:,1]<=.15*max(stats[i,1],1e-6))]
        pp=match(tr[(w[tr]==w[i])&(s[tr]==s[i])&(c[tr]!=c[i])]);cross=pp[src[pp]!=src[i]]
        pp=cross if len(cross) else pp
        nn=match(q[(w[q]!=w[i])&(s[q]==s[i])&(c[q]==c[i])])
        if len(pp) and len(nn):pools[int(i)]=(pp,nn)
    eligible=set(pools);rng=np.random.default_rng(89);best=[]
    for trial in range(200):
        order=np.array(list(pools));rng.shuffle(order)
        if trial%2==0:order=np.array(sorted(order,key=lambda i:len(pools[i][1])))
        used=set();pairs=[]
        for i in order:
            if int(i) in used:continue
            pp,nn=pools[int(i)];pp=[int(j) for j in pp if int(j) not in used];nn=[int(j) for j in nn if int(j) not in used]
            if not pp or not nn:continue
            # Preserve eligible anchors where possible; all choices feature-blind.
            preferred=[j for j in nn if j not in eligible];nn=preferred or nn
            p=int(rng.choice(pp));n=int(rng.choice(nn));pairs.append([int(i),p,n]);used.update((int(i),p,n))
        if len(pairs)>len(best):best=pairs
    pairs=np.array(best,int);assert len(set(pairs.ravel()))==pairs.size
    np.savez_compressed('assets/source_disjoint_expanded.npz',pairs=pairs)
    report={'n':len(pairs),'eligible_anchors':len(eligible),'n_characters':len(set(c[pairs[:,0]])),
        'n_sources':len(set(src[pairs[:,0]])),'selection':'200 seeded feature-blind greedy trials, maximum cardinality',
        'note':'Auxiliary protocol: different triplets from original 196; same source pool/constraints, no repeated images, no claim of independent artworks.'}
    # Exact upper bound: relax unique positives, but require anchor/negative
    # images to be disjoint. A maximum graph matching bounds all triplet packs.
    import networkx as nx
    graph=nx.Graph()
    for anchor,(_,negatives) in pools.items():
        graph.add_edges_from((int(anchor),int(n)) for n in negatives)
    bound=len(nx.max_weight_matching(graph,maxcardinality=True))
    report['exact_anchor_negative_matching_upper_bound']=bound
    report['upper_bound_note']='Upper bound under the fixed query pool/matching rules, even before unique-positive constraints.'
    Path('assets/source_disjoint_expanded.json').write_text(json.dumps(report,indent=2))
    print('SOURCE_DISJOINT_EXPANDED',json.dumps(report),flush=True)


if __name__=='__main__':main()
