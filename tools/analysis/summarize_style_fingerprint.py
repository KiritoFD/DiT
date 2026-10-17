"""Export compact audit tables, QC and paired improvements on identical triplets."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.analysis.style_fingerprint_features import matched_pairs
from tools.analysis.run_style_fingerprint_audit import interval, pair_scores


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True)
    p.add_argument('--skip-plots',action='store_true')
    p.add_argument('--split-root',required=True);a=p.parse_args();out=Path(a.out)
    report=json.loads((out/'results.json').read_text())
    d=np.load('scratch/disc_lat_full.npz',allow_pickle=True);f=np.load(out/'features.npz')
    w,s,c,source=[d[k].astype(str) for k in ('writer','script','char','work')]
    split=np.load(Path(a.split_root)/'split_indices.npz')
    stats=np.load(Path(a.split_root)/'ink_width_stats.npz')['stats']
    rows=[r for r in csv.DictReader(open('exp-std/csv/train.csv',encoding='utf-8-sig')) if not (r.get('aug') or '').strip()]
    paths=sorted({r['std_path'] for r in rows});lookup={v:i for i,v in enumerate(paths)};gi=np.array([lookup[r['std_path']] for r in rows])
    keys=[k[5:] for k in f.files if k.startswith('real_') and k not in ('real_qc','real_latent_std')]
    raw={k:f['real_'+k] for k in keys}
    ref={k:(f['real_latent_std'] if k=='latent' else f['std_'+k][gi]) for k in keys}
    good=np.logical_and.reduce([np.isfinite(raw[k]).all(1)&np.isfinite(ref[k]).all(1) for k in keys])
    supplemental={'n_common':int(good.sum()),'feature_coverage':{},'paired_improvements':{},'qc':{},'source_identifiability':{},'family_full_valid_matched':{}}
    for k in keys:supplemental['feature_coverage'][k]={'n_valid':int((np.isfinite(raw[k]).all(1)&np.isfinite(ref[k]).all(1)).sum()),'dimension':raw[k].shape[1]}
    for j,label in [(0,'ink_fraction'),(1,'median_width'),(2,'valid_oriented_skeleton_points'),(3,'intermediate_gray_fraction')]:
        x=f['real_qc'][:,j];x=x[np.isfinite(x)]
        supplemental['qc'][label]=dict(zip(['p10','median','p90'],np.quantile(x,[.1,.5,.9]).tolist()))
    for script in sorted(set(s)):
        ix=np.flatnonzero(s==script)
        # Writer-character connected components: effects across disconnected
        # blocks cannot be interpreted as common-character writer contrasts.
        writers=sorted(set(w[ix]));adj={x:set() for x in writers}
        for char in set(c[ix]):
            ws=set(w[ix[c[ix]==char]])
            for writer in ws:adj[writer]|=ws
        components=[];todo=set(writers)
        while todo:
            seen=set();stack=[min(todo)]
            while stack:
                node=stack.pop()
                if node in seen:continue
                seen.add(node);stack.extend(adj[node]-seen)
            todo-=seen;components.append(sorted(seen))
        srcwriters={src:set(w[ix[source[ix]==src]]) for src in set(source[ix])}
        supplemental['source_identifiability'][script]={'writer_character_components':components,
            'source_count':len(srcwriters),'sources_shared_by_writers':sum(len(x)>1 for x in srcwriters.values()),
            'note':'Single-source writers and writer-nested sources prevent independent source-adjusted writer attribution.'}
    delta={k:raw[k]-ref[k] for k in keys}
    delta['geometry_combo']=np.concatenate([delta[k] for k in ('quill','hinge','profile')],1)
    for protocol,qkey in [('character_holdout','test_char'),('source_holdout','test_source')]:
        tr=split['train'];q=split[qkey];tr=tr[good[tr]];q=q[good[q]]
        pairs,_=matched_pairs(q,tr,w,s,c,source,stats)
        outcomes={}
        for k,x in delta.items():
            x=(x-x[tr].mean(0))/np.maximum(x[tr].std(0),.05)
            aa,pp,nn=x[pairs[:,0]],x[pairs[:,1]],x[pairs[:,2]]
            margin=np.mean((aa-nn)**2,1)-np.mean((aa-pp)**2,1)
            outcomes[k]=(margin>0).astype(float)+.5*(margin==0)
        for baseline in ('ink','width'):
            diff=outcomes['geometry_combo']-outcomes[baseline]
            supplemental['paired_improvements'][protocol+'__geometry_minus_'+baseline]={
                'by_char':interval(diff,c[pairs[:,0]]),'by_source':interval(diff,source[pairs[:,0]])}
        supplemental['family_full_valid_matched'][protocol]={}
        for k in keys:
            valid=np.isfinite(raw[k]).all(1)&np.isfinite(ref[k]).all(1)
            ft=split['train'][valid[split['train']]];fq=split[qkey][valid[split[qkey]]]
            fp,_=matched_pairs(fq,ft,w,s,c,source,stats)
            x=delta[k];x=(x-x[ft].mean(0))/np.maximum(x[ft].std(0),.05)
            supplemental['family_full_valid_matched'][protocol][k]=pair_scores(x,fp,c,source)
    (out/'supplemental.json').write_text(json.dumps(supplemental,ensure_ascii=False,indent=2),encoding='utf-8')
    columns=['protocol','family','mode','n_common_query','matched_n','matched_2afc','char_ci_low','char_ci_high','source_ci_low','source_ci_high','crosschar_2afc','prototype_top1','samechar_2afc','real_std_auc']
    with (out/'ranking.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,columns);writer.writeheader()
        for protocol,families in report['results'].items():
            for family,modes in families.items():
                for mode,result in modes.items():
                    ch=result['matched']['by_char'];src=result['matched']['by_source'];ci=ch['ci95'] or [None,None];si=src['ci95'] or [None,None]
                    writer.writerow(dict(zip(columns,[protocol,family,mode,result['n_query'],ch['n'],ch['accuracy'],*ci,*si,result['crosschar']['by_char']['accuracy'],result['prototype_top1']['accuracy'],result['samechar_prototype_2afc']['accuracy'],result['real_std_auc_script_macro']])))
    # High-dimensional Quill maps expose direction/width effects without the
    # cancellation produced by averaging signed histogram components.
    slots=sorted(set(zip(w,s)));tr=split['train'];tr=tr[good[tr]]
    scale=np.maximum(raw['quill'][tr,:96].std(0),.001)
    offset=[];residual=[]
    for writer,script in slots:
        ix=np.flatnonzero((w==writer)&(s==script)&good);peers=np.flatnonzero((s==script)&good)
        offset.append((raw['quill'][ix,:96].mean(0)-raw['quill'][peers,:96].mean(0))/scale)
        residual.append((delta['quill'][ix,:96].mean(0)-delta['quill'][peers,:96].mean(0))/scale)
    np.savez_compressed(out/'heatmap_data.npz',offset=offset,residual=residual,slots=np.array(slots))
    if a.skip_plots:
        print('FINGERPRINT_SUMMARY_COMPLETE',out,flush=True)
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for name,values,title in [('quill_writer_heatmap',offset,'Writer offset from script mean / Quill-like absolute-width bins'),('quill_residual_heatmap',residual,'Writer-specific GT-STD residual after removing script mean')]:
        fig,ax=plt.subplots(figsize=(15,8));v=np.asarray(values);limit=max(float(np.quantile(np.abs(v),.98)),.1)
        im=ax.imshow(v,aspect='auto',cmap='coolwarm',vmin=-limit,vmax=limit);fig.colorbar(im,ax=ax)
        ax.set_xticks(np.arange(0,96,8),[f'{i*15}deg' for i in range(12)])
        ax.set_yticks(range(len(slots)),[f'S{i:02d}' for i in range(len(slots))]);ax.set_title(title+' (descriptive, confounded)')
        fig.tight_layout();fig.savefig(out/(name+'.png'),dpi=150);plt.close(fig)
    print(json.dumps(supplemental,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
