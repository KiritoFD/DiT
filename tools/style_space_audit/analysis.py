"""GT-only PCA, full-space clustering, cross-character retrieval and non-Gaussian distances."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score
from tools.three_day.contracts import save


def clustering(f,rows):
    result={}
    for script in sorted({r['script'] for r in rows}):
        ix=[i for i,r in enumerate(rows) if r['script']==script];labels=[rows[i]['writer'] for i in ix];k=len(set(labels))
        if k<2 or len(ix)<2*k:
            result[script]=dict(n=len(ix),writers=k,ari=None,reason='Insufficient distinct supported writers');continue
        pred=KMeans(k,random_state=42,n_init=20).fit_predict(f[ix])
        result[script]=dict(n=len(ix),writers=k,ari=float(adjusted_rand_score(labels,pred)),nmi=float(normalized_mutual_info_score(labels,pred)))
    return result


def mmd(a,b,bandwidth):
    def kernel(x,y):return np.exp(-np.maximum(0,(x*x).sum(1)[:,None]+(y*y).sum(1)[None,:]-2*x@y.T)/(2*bandwidth))
    aa=kernel(a,a);bb=kernel(b,b);ab=kernel(a,b)
    return float((aa.sum()-np.trace(aa))/(len(a)*(len(a)-1))+(bb.sum()-np.trace(bb))/(len(b)*(len(b)-1))-2*ab.mean())


def score(f,rows,rf,reference,bandwidth):
    labels=np.array([r['writer'] for r in reference]);scripts=np.array([r['script'] for r in reference]);chars=np.array([r['char'] for r in reference])
    per=[];groups=defaultdict(list)
    for i,r in enumerate(rows):
        legal=(scripts==r['script'])&(chars!=r['char']);slots=sorted(set(labels[legal]));target=r['writer']
        if target not in slots or len(slots)<2:continue
        proto=np.stack([rf[legal&(labels==w)].mean(0) for w in slots]);proto/=np.linalg.norm(proto,axis=1,keepdims=True).clip(1e-8)
        sims=proto@f[i];ti=slots.index(target);others=np.delete(sims,ti)
        per.append(dict(idx=i,writer=target,char=r['char'],script=r['script'],correct=bool(slots[sims.argmax()]==target),margin=float(sims[ti]-others.max())))
        groups[target,r['script']].append(i)
    distances=[]
    for (w,s),ids in groups.items():
        ii=np.flatnonzero((labels==w)&(scripts==s))
        if len(ids)<2 or len(ii)<2:continue
        a=f[ids];b=rf[ii]
        spread=float(((b-b.mean(0))**2).sum(1).mean())
        distances.append(dict(writer=w,script=s,n_gen=len(a),n_ref=len(b),mmd2_unbiased=mmd(a,b,bandwidth),
            centroid_l2=float(np.linalg.norm(a.mean(0)-b.mean(0))),variance_ratio=float(((a-a.mean(0))**2).sum(1).mean()/max(spread,1e-8)),
            distance_over_gt_spread=float(np.linalg.norm(a.mean(0)-b.mean(0))/np.sqrt(max(spread,1e-8)))))
    return dict(n=len(rows),n_eligible=len(per),crosschar_top1=float(np.mean([r['correct'] for r in per])) if per else None,
        margin=float(np.mean([r['margin'] for r in per])) if per else None,
        distribution=distances,clustering=clustering(f,rows),per_image=per)


def main(root):
    root=Path(root);real=json.loads((root/'real_rows.json').read_text());rows=json.loads((root/'rows.json').read_text())
    old=json.loads((root/'original_GT_indices.json').read_text());identity=json.loads((root/'teacher_identity.json').read_text())
    from src.eval.style.bank import work_id
    train_chars={r['char'] for r in real if not r['source_heldout']}
    for r in rows:
        r['work']=work_id(r.get('src_image_path',''),r['image_path'])
        r['encoder_source_heldout']=r['work'] in identity['holdout_works']
        r['encoder_character_heldout']=r['char'] not in train_chars
    report=dict(teacher=identity,protocol='PCA fit GT only; all reported clustering uses original L2-normalized 256D, never PCA2D',
        background='white: grayscale, removes first .08 ink intensity; ink_match uses fractional morphology to match paired GT mass and changes geometry',
        limitations=['Reference work labels are directory proxies; original training CSV/hash not stored in checkpoint.',
                    'ARI measures label clustering, not distribution realism. Unbiased MMD2 may be negative.',
                    'P2/Deep/Qwen targets matched exactly; Deep uses P2 STD/noise and may see a condition-domain shift.'],variants={})
    overlap=root/'Deep114_native_training_overlap.json'
    if overlap.exists():report['generator_overlap']={'Deep114':json.loads(overlap.read_text())}
    held=[i for i,r in enumerate(real) if r['dedup_source_heldout']]
    dual=[i for i in held if real[i]['character_heldout']]
    report['GT_protocol_counts']={name:dict(n=len(ids),slots=dict(Counter(r['writer']+'_'+r['script'] for r in [real[i] for i in ids])))
        for name,ids in [('original_first120',old),('dedup_source_holdout',held),('source_and_global_character_holdout',dual)]}
    report['paired100_encoder_status']=dict(n=len(rows),source_heldout=sum(r['encoder_source_heldout'] for r in rows),
        global_character_heldout=sum(r['encoder_character_heldout'] for r in rows),
        both=sum(r['encoder_source_heldout'] and r['encoder_character_heldout'] for r in rows))
    save(root/'paired_rows_audited.json',rows)
    for mode in ('raw','white','ink_match'):
        realmode='white' if mode=='ink_match' else mode
        f=np.load(root/'features'/('REAL_'+realmode+'.npz'))['embedding']
        refs={}
        # Class-balanced deterministic GT anchors for plots; no generated image participates in fitting.
        groups=defaultdict(list)
        for i,r in enumerate(real):groups[r['writer'],r['script']].append(i)
        fit=[];rng=np.random.default_rng(42)
        for key,ids in sorted(groups.items()):fit.extend(rng.choice(ids,min(len(ids),120),replace=False).tolist())
        pca=PCA(2,svd_solver='full').fit(f[fit]);np.savez(root/(mode+'_pca.npz'),mean=pca.mean_,components=pca.components_,gt_indices=fit)
        bandwidth=float(np.median(np.maximum(1e-8,2-2*(f[fit[:500]]@f[fit[:500]].T))))
        for protocol,ids in [('GT_source_holdout',held),('GT_source_character_holdout',dual),('GT_all_exploratory',fit)]:
            # Bound reference size per real slot; each query excludes its own character at score time.
            selected=[];counts=Counter()
            for i in ids:
                key=real[i]['writer'],real[i]['script']
                if counts[key]<120:selected.append(i);counts[key]+=1
            refs[protocol]=(f[selected],[real[i] for i in selected])
        refs['GT_matched100']=(np.load(root/'features'/('GT_'+mode+'.npz'))['embedding'],rows)
        result={'GT_clustering':{name:clustering(f[ids],[real[i] for i in ids]) for name,ids in [('original_first120',old),('source_holdout',held),('source_character_holdout',dual)]},
                'pca_gt_only_evr':pca.explained_variance_ratio_.tolist(),'models':{},
                'controlled_writer_scopes':controlled_scopes(f,real,old,held),
                'real_head_accuracy':head_accuracy(np.load(root/'features'/('REAL_'+realmode+'.npz'))['logits'],real,identity['slots'],old,held,dual)}
        for name in ('STD','GT','VAE_GT','Deep114','Deep114_native','P2_CFG1','P2_CFG1.6','Qwen'):
            ff=np.load(root/'features'/(name+'_'+mode+'.npz'))['embedding']
            result['models'][name]={protocol:score(ff,rows,rf,rr,bandwidth) for protocol,(rf,rr) in refs.items() if len(rr)>=2}
            logits=np.load(root/'features'/(name+'_'+mode+'.npz'))['logits'];pred=logits.argmax(1)
            slots=[tuple(k) for k in identity['slots']];truth=np.array([slots.index((r['writer'],r['script'])) for r in rows])
            result['models'][name]['classifier_head']=dict(target_slot_top1=float((pred==truth).mean()),
                std_predicted_fraction=float(np.mean([slots[p][0].startswith('std') for p in pred])))
            np.save(root/'features'/(name+'_'+mode+'_coords.npy'),pca.transform(ff))
        from tools.three_day.audits.metrics import interval
        aa=result['models']['P2_CFG1']['GT_matched100']['per_image'];bb=result['models']['P2_CFG1.6']['GT_matched100']['per_image']
        if [r['idx'] for r in aa]!=[r['idx'] for r in bb]:raise ValueError('Paired eligible sets changed')
        result['CFG_paired_delta']={key:interval([float(b[key])-float(a[key]) for a,b in zip(aa,bb)],[r['char'] for r in aa]) for key in ('correct','margin')}
        gt=np.load(root/'features'/('GT_'+mode+'.npz'))['embedding'];std=np.load(root/'features'/('STD_'+mode+'.npz'))['embedding']
        result['GT_vs_STD_proximity']={}
        for name in ('Deep114_native','P2_CFG1','P2_CFG1.6','Qwen'):
            ff=np.load(root/'features'/(name+'_'+mode+'.npz'))['embedding'];dg=np.linalg.norm(ff-gt,axis=1);ds=np.linalg.norm(ff-std,axis=1)
            result['GT_vs_STD_proximity'][name]=dict(mean_distance_GT=float(dg.mean()),mean_distance_STD=float(ds.mean()),
                closer_to_STD=interval(ds<dg,[r['char'] for r in rows]))
        report['variants'][mode]=result;save(root/'results.json',report)
    direction(root,real)
    dino_audit(root,real,old)
    render(root,report,real)
    print('STYLE_SPACE_ANALYSIS_DONE',flush=True)


def dino_audit(root,real,old):
    from tools.style_space_audit.data import NAMES
    from tools.three_day.contracts import sha
    path=Path('assets/dino_feat_top10_g.npz');cache=np.load(path,allow_pickle=True)
    ww=np.array([NAMES[int(w)] for w in cache['calligs']]);ss=np.array([{0:'楷',3:'行',4:'隶'}[int(s)] for s in cache['scripts']])
    order_matches=len(real)==len(ww) and all(r['writer']==w and r['script']==s for r,w,s in zip(real,ww,ss))
    result=dict(cache=str(path),sha256=sha(path),writer_script_sequence_matches_real_bank=order_matches,
        caveat='Cache does not contain image hashes; matching writer/script sequence alone does not prove exact image identity.',scopes={})
    for script,writers,name in [('楷',['颜真卿','欧阳询','柳公权','褚遂良','赵孟頫'],'kai5'),('行',['王羲之','米芾','苏轼','文徵明'],'xing4')]:
        ids=[]
        for w in writers:ids.extend(np.flatnonzero((ww==w)&(ss==script))[:120].tolist())
        labels=ww[ids];f=cache['feat'][ids].astype(float)
        item={}
        for normalization in ('cached','L2'):
            z=f if normalization=='cached' else f/np.linalg.norm(f,axis=1,keepdims=True).clip(1e-8)
            pred=KMeans(len(writers),random_state=42,n_init=20).fit_predict(z)
            item[normalization]=dict(n=len(ids),ari=float(adjusted_rand_score(labels,pred)),nmi=float(normalized_mutual_info_score(labels,pred)))
        result['scopes'][name]=item
    save(root/'dino_recomputed.json',result)


def head_accuracy(logits,real,slots,old,held,dual):
    keys=[tuple(k) for k in slots];truth=np.array([keys.index((r['writer'],r['script'])) for r in real]);pred=logits.argmax(1)
    conditioned=np.zeros(len(real),dtype=int)
    for script in sorted({r['script'] for r in real}):
        candidates=np.array([i for i,k in enumerate(keys) if k[1]==script and not k[0].startswith('std')])
        ids=np.array([i for i,r in enumerate(real) if r['script']==script]);conditioned[ids]=candidates[logits[ids][:,candidates].argmax(1)]
    return {name:dict(n=len(ids),slot_CE_top1=float((pred[ids]==truth[ids]).mean()) if ids else None,
        script_conditioned_real_writer_top1=float((conditioned[ids]==truth[ids]).mean()) if ids else None)
        for name,ids in [('original_first120',old),('source_holdout',held),('source_character_holdout',dual)]}


def controlled_scopes(f,real,old,held):
    scopes={'kai5':('楷',['颜真卿','欧阳询','柳公权','褚遂良','赵孟頫']),
            'xing4':('行',['王羲之','米芾','苏轼','文徵明'])};result={}
    for scope,(script,writers) in scopes.items():
        a=[i for i in old if real[i]['writer'] in writers and real[i]['script']==script]
        b=[i for i in held if real[i]['writer'] in writers and real[i]['script']==script]
        counts=Counter(real[i]['writer'] for i in b);common=[w for w in writers if counts[w]>=2]
        n=min([counts[w] for w in common]+[60]);rng=np.random.default_rng(42);balanced={}
        for name,pool in [('original',a),('source_holdout',b)]:
            per=[]
            for seed in range(10):
                ids=[]
                for w in common:
                    choices=[i for i in pool if real[i]['writer']==w]
                    ids.extend(rng.choice(choices,n,replace=False).tolist())
                if len(common)>=2:
                    pred=KMeans(len(common),random_state=seed,n_init=10).fit_predict(f[ids]);per.append(float(adjusted_rand_score([real[i]['writer'] for i in ids],pred)))
            balanced[name]=dict(n_per_writer=n,n_writers=len(common),ari_mean=float(np.mean(per)) if per else None,
                ari_range=[min(per),max(per)] if per else None)
        result[scope]=dict(requested_writers=writers,missing_source_heldout=[w for w in writers if w not in counts],
            original=clustering(f[a],[real[i] for i in a]),source_heldout=clustering(f[b],[real[i] for i in b]),
            balanced_common_writers=common,balanced_comparison=balanced)
    return result


def direction(root,real):
    rows=json.loads((root/'swap_rows.json').read_text());groups=defaultdict(dict)
    for i,r in enumerate(rows):groups[r['script'],r['char']][r['writer']]=i
    result={}
    from tools.three_day.audits.metrics import interval
    for mode in ('raw','white'):
        f=np.load(root/'features'/('REAL_'+mode+'.npz'))['embedding']
        g=np.load(root/'features'/('P2_swap_'+mode+'.npz'))['embedding']
        for protocol in ('all_GT_exploratory','source_heldout_GT'):
            ref=defaultdict(list)
            for i,r in enumerate(real):
                if protocol=='source_heldout_GT' and not r['dedup_source_heldout']:continue
                ref[r['writer'],r['script'],r['char']].append(i)
            pairs=[]
            for (script,char),writers in groups.items():
                ww=sorted(writers)
                for j,a in enumerate(ww):
                    for b in ww[j+1:]:
                        ia=ref[a,script,char];ib=ref[b,script,char]
                        if not ia or not ib:continue
                        true=f[ib].mean(0)-f[ia].mean(0);pred=g[writers[b]]-g[writers[a]]
                        denominator=np.linalg.norm(true)*np.linalg.norm(pred)
                        if denominator<1e-8:continue
                        pairs.append(dict(char=char,script=script,a=a,b=b,cosine=float(true@pred/denominator),
                            n_gt_a=len(ia),n_gt_b=len(ib)))
            result[mode+'_'+protocol]=dict(n_pairs=len(pairs),n_chars=len({p['char'] for p in pairs}),
                cosine=interval([p['cosine'] for p in pairs],[p['char'] for p in pairs]),
                positive_direction=interval([p['cosine']>0 for p in pairs],[p['char'] for p in pairs]),per_pair=pairs)
    save(root/'writer_direction.json',result)


def render(root,report,real):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname='tools/fonts/simkai.ttf') if Path('tools/fonts/simkai.ttf').exists() else None
    models=['GT','VAE_GT','Deep114_native','P2_CFG1','P2_CFG1.6','Qwen'];rows=json.loads((root/'rows.json').read_text())
    for script in ('楷','行','隶'):
        fig,axes=plt.subplots(2,3,figsize=(15,9));coords=[]
        selected=[i for i,r in enumerate(rows) if r['script']==script]
        for name in models:coords.append(np.load(root/'features'/(name+'_raw_coords.npy'))[selected])
        if not selected:plt.close(fig);continue
        all_points=np.concatenate(coords);lo=all_points.min(0)-.05;hi=all_points.max(0)+.05
        writers=sorted({rows[i]['writer'] for i in selected});colors={w:plt.get_cmap('tab10')(j) for j,w in enumerate(writers)}
        for ax,name,points in zip(axes.flat,models,coords):
            for w in writers:
                ix=[j for j,i in enumerate(selected) if rows[i]['writer']==w]
                ax.scatter(points[ix,0],points[ix,1],s=25,color=colors[w],label=w,alpha=.75)
            ax.set_title(name);ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.grid(alpha=.2)
        axes.flat[-1].legend(prop=font,fontsize=8)
        fig.suptitle('Matched targets / identical GT-only PCA / '+script,fontproperties=font)
        fig.tight_layout();fig.savefig(root/('matched_pca_'+script+'.png'),dpi=160);plt.close(fig)
    # Original high-ARI selection and saved-source-holdout selection in the SAME GT-only projection.
    f=np.load(root/'features/REAL_raw.npz')['embedding'];p=np.load(root/'raw_pca.npz');xy=(f-p['mean'])@p['components'].T
    old=json.loads((root/'original_GT_indices.json').read_text())
    held=[i for i,r in enumerate(real) if r['dedup_source_heldout']]
    fig,axes=plt.subplots(1,2,figsize=(14,6));writers=sorted({r['writer'] for r in real})
    for ax,ids,title in zip(axes,[old,held],['Original first-120 selection','Saved-source holdout, pixel deduplicated']):
        for j,w in enumerate(writers):
            ix=[i for i in ids if real[i]['writer']==w]
            if ix:ax.scatter(xy[ix,0],xy[ix,1],s=6,color=plt.get_cmap('tab10')(j),label=w,alpha=.4)
        ax.set_title(title);ax.grid(alpha=.2)
    axes[-1].legend(prop=font,fontsize=8);fig.tight_layout();fig.savefig(root/'GT_overlap_control.png',dpi=160);plt.close(fig)
    from PIL import Image,ImageDraw
    indices=[];seen=set()
    for i,r in enumerate(rows):
        if r['writer'] not in seen:indices.append(i);seen.add(r['writer'])
    q=json.loads((root/'qwen_pairing.json').read_text())['paths'];names=['STD','GT','VAE_GT','Deep114_native','Deep114_shared','P2_CFG1','P2_CFG1.6','Qwen']
    canvas=Image.new('RGB',(8*128,26+len(indices)*128),'#eeeeee');d=ImageDraw.Draw(canvas)
    for j,name in enumerate(names):d.text((128*j+3,6),name,fill='black')
    for row,i in enumerate(indices):
        for j,name in enumerate(names):
            path=Path(q[i]) if name=='Qwen' else root/('Deep114' if name=='Deep114_shared' else name)/f'g{i}.png'
            canvas.paste(Image.open(path).convert('RGB').resize((128,128)),(j*128,26+row*128))
    canvas.save(root/'matched_examples.png');save(root/'poster_indices.json',indices)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();main(a.root)
