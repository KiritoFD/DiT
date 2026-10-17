"""Train-only conditional scatter probe. No neural training or DiT changes."""
import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.writer_quartets import WriterQuartets
from tools.analysis.allograph_stats import l2, make_trials, measure
from tools.analysis.accept_writer_style_pixel import grouped


def fit_projection(x, blocks, rank=128, output=16):
    """x and blocks contain TRAINING rows only; shrink denominator scatter."""
    mean=x.mean(0)
    _, _, basis=torch.pca_lowrank(x-mean, q=min(rank, min(x.shape)-1), center=False, niter=3)
    y=(x-mean)@basis
    q=y[blocks]
    dw=torch.cat((q[:,0]-q[:,2],q[:,1]-q[:,3]))
    dc=torch.cat((q[:,0]-q[:,1],q[:,2]-q[:,3]))
    sw=dw.T@dw/len(dw);sc=dc.T@dc/len(dc)
    reg=.8*sc+.2*sc.trace()/len(sc)*torch.eye(len(sc),device=x.device)
    reg=reg+1e-6*torch.eye(len(sc),device=x.device)
    chol=torch.linalg.cholesky(reg)
    inv=torch.linalg.solve_triangular(chol,torch.eye(len(sc),device=x.device),upper=False)
    vals,vec=torch.linalg.eigh(inv@sw@inv.T)
    w=inv.T@vec[:,-min(output,len(vals)):]
    return mean,basis,w,vals


def directions(x,data,key):
    """Training writer-pair directions vs heldout chars; source uses LOO chars."""
    train=data['splits']['train'];test=data['splits'][key]
    lab=data['labels'];chars=data['char'];rows=[];pair_deltas={}
    def bank(ids,w):
        ids=ids[lab[ids]==w]
        return {c:x[ids[chars[ids]==c]].mean(0) for c in set(chars[ids])}
    for a,b in combinations(range(len(data['slots'])),2):
        if data['slots'][a][1]!=data['slots'][b][1]:continue
        aa,ab,ba,bb=bank(train,a),bank(train,b),bank(test,a),bank(test,b)
        ca=sorted(set(aa)&set(ab));cb=sorted(set(ba)&set(bb))
        if min(len(ca),len(cb))<4:continue
        td={c:aa[c]-ab[c] for c in ca};ed=l2(np.stack([ba[c]-bb[c] for c in cb]))
        total=np.stack(list(td.values())).sum(0)
        prototypes={c:l2((total-td.get(c,0))/(len(td)-int(c in td))) for c in cb}
        pair_deltas[a,b]=(prototypes,dict(zip(cb,ed)))
        score=[]
        for c,d in zip(cb,ed):
            available=[v for cc,v in td.items() if cc!=c]
            if available:score.append(float(l2(np.mean(available,axis=0))@d))
        coherence=float((np.square(ed.sum(0)).sum()-len(ed))/(len(ed)*(len(ed)-1)))
        rng=np.random.default_rng(89+a*31+b)
        null=[]
        for _ in range(200):
            signed=ed*rng.choice([-1,1],len(ed))[:,None]
            null.append(float((np.square(signed.sum(0)).sum()-len(ed))/(len(ed)*(len(ed)-1))))
        rows.append({'a':a,'b':b,'train_chars':len(ca),'test_chars':len(cb),
                     'train_to_test_cos':float(np.mean(score)),'crosschar_cos':coherence,
                     'within_pair_identity_swap_null95':np.quantile(null,[.025,.975]).tolist()})
    for r in rows:
        a,b=r['a'],r['b'];prototypes,ed=pair_deltas[a,b];controls=[];own=[]
        for (u,v),(_,other) in pair_deltas.items():
            if {a,b}&{u,v} or data['slots'][a][1]!=data['slots'][u][1]:continue
            for c in sorted(set(ed)&set(other)):
                direction=prototypes[c]
                controls.append(float(direction@other[c]));own.append(float(direction@ed[c]))
        r['disjoint_writer_pair_samechar_control_cos']=float(np.mean(controls)) if controls else None
        r['own_minus_disjoint_control_cos']=float(np.mean(np.array(own)-controls)) if controls else None
        r['n_control_comparisons']=len(controls)
    means=np.array([r['train_to_test_cos'] for r in rows])
    rng=np.random.default_rng(89)
    ci=np.quantile(means[rng.integers(len(means),size=(1000,len(means)))].mean(1),[.025,.975]).tolist() if len(means) else None
    return {'pairs':rows,'pair_bootstrap_train_to_test_cos_ci95':ci,'mean_pair_train_to_test_cos':float(np.mean(means)) if rows else None,
            'mean_pair_crosschar_cos':float(np.mean([r['crosschar_cos'] for r in rows])) if rows else None,
            'note':'Null independently swaps writer identities per character; not a global writer permutation. Pair coverage >=4 train and test shared chars. Source directions exclude test character from train prototype.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--pixel',required=True);p.add_argument('--out',required=True)
    p.add_argument('--directions-only',action='store_true',help='Supplement cached results with direction controls without refitting projections')
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(89);torch.cuda.set_per_process_memory_fraction(.12)
    data=torch.load(a.data,map_location='cpu',weights_only=False)
    train=np.asarray(data['splits']['train']);z=data['z'].float()
    sampler=WriterQuartets(train,data['labels'],data['si'],data['char'],data['source'])
    ids,coverage=sampler.sample(32768);blocks=ids.reshape(-1,4)
    trials={}
    if not a.directions_only:
        for key in ('test_char','test_source'):
            for k in (1,4,16,32):trials[key,k]=make_trials(data,k,key=='test_source',repeats=30)
    report={'config':{'pca':128,'projection':16,'shrinkage':.2,'seed':89,'fit':'train only, separately within script','train_indices_sha256':hashlib.sha256(train.tobytes()).hexdigest(),'source':'directory proxies, not verified artworks'},'quartets':coverage,'methods':{}}
    artifacts=torch.load(out/'projections.pt',map_location='cpu',weights_only=False) if a.directions_only else {}
    if a.directions_only:
        expected_train_hash=report['config']['train_indices_sha256']
        report=json.loads((out/'results.json').read_text(encoding='utf-8'))
        assert report['config']['train_indices_sha256']==expected_train_hash
        assert len(artifacts)==9,'All three scales and three scripts must already be fitted'
    def evaluate(name,x,delta=True):
        if a.directions_only:
            for key in ('test_char','test_source'):report['methods'][name][key]['directions']=directions(x,data,key)
            print('DIRECTION_CONTROLS_COMPLETE',name,flush=True)
            return
        normalized=l2(x);r={}
        for key in ('test_char','test_source'):
            r[key]={'matched':grouped(normalized,data['matched_pairs'][key],data),'aggregation':{}}
            if delta:r[key]['directions']=directions(x,data,key)
            for k in (1,4,16,32):
                ts,cv=trials[key,k];r[key]['aggregation'][str(k)]=measure(x,ts,'shape')
            # Fixed 32-character-eligible pair cohort: aggregation-size comparison.
            eligible={(t['w'],t['v']) for t in trials[key,32][0]}
            r[key]['common32_cohort']={str(k):measure(x,[t for t in trials[key,k][0] if (t['w'],t['v']) in eligible],'shape') for k in (1,4,16,32)}
        report['methods'][name]=r
        (out/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print('PROBE_COMPLETE',name,{k:r[k]['matched']['by_char']['accuracy'] for k in r},flush=True)
    for size in (32,16,8):
        x=(z if size==32 else F.adaptive_avg_pool2d(z,size)).flatten(1).numpy()
        evaluate(f'raw{size}',x)
        pc=np.zeros((len(z),128),np.float32);proj=np.zeros((len(z),16),np.float32)
        for script in sorted(set(data['si'])):
            fitids=train[data['si'][train]==script];allids=np.where(data['si']==script)[0]
            lookup=np.full(len(z),-1);lookup[fitids]=np.arange(len(fitids))
            selected=blocks[data['si'][blocks[:,0]]==script]
            local=lookup[selected];assert (local>=0).all()
            artifact_key=f'{size}_{script}'
            if artifact_key in artifacts:
                mean,basis,w,vals=(artifacts[artifact_key][k].cuda() for k in ('mean','basis','w','eigenvalues'))
            else:
                mean,basis,w,vals=fit_projection(torch.from_numpy(x[fitids]).cuda(),torch.from_numpy(local).cuda())
            with torch.no_grad():y=(torch.from_numpy(x[allids]).cuda()-mean)@basis;yy=y@w
            pc[allids]=y.cpu().numpy();proj[allids]=yy.cpu().numpy()
            artifacts[f'{size}_{script}']={k:v.cpu() for k,v in zip(('mean','basis','w','eigenvalues'),(mean,basis,w,vals))}
            del mean,basis,w,vals,y,yy;torch.cuda.empty_cache()
        evaluate(f'pca{size}',pc);evaluate(f'conditional{size}',proj)
        torch.save(artifacts,out/'projections.pt')
    if not a.directions_only:
        pixel=np.load(a.pixel)
        evaluate('pixel_D1_raw',pixel['raw'],False);evaluate('pixel_D1_reconstructed_mode',pixel['reconstructed'],False)
    report['limitations']=['No heldout hyperparameter selection; fixed PCA128/projection16/shrinkage0.2.', 'Source labels are proxies; source confounding is not eliminated.', 'Conditional scatter is a probe, not a complete nonlinear style metric.', 'Pair-coherence null is identity swap per character; no claim of global-permutation significance.', 'Pixel reconstructed reference is posterior-mode full cache; exact shard bridge is recorded separately.']
    (out/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
