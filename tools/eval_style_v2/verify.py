"""Phase 1+2 orchestrator: bank features, real cross-char validation, panel
verification, throughput measurement. CPU only; nice/limits set by the caller.

Outputs:
  real_encoder_validation.json  - per feature x K: AUC/EER/mAP + permutation + CI
  writer_verification.csv       - real + panel rows
  throughput.json               - Hinge/Quill extraction rate measured on real images
"""
import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
from .common import load_real_rows,save_json,read_json,FEATURE_VERSION,PREPROCESS
from . import features_hq as hq
from . import features_eg as eg
from . import metrics as M

FEATURES=('E-G','hinge_s','hinge_l','quill','hq_concat')
HQ_KEYS=('hinge_s','hinge_l','quill')


def bank_hq(bank_rows,machine='48'):
    cache=Path(f'assets/eval_style_v2_cache/{machine}/hq_bank.npz')
    if cache.is_file():
        z=np.load(cache)
        if z['version'].item()==FEATURE_VERSION and z['rows_sha']==json.dumps([r.get('pixel_sha256') for r in bank_rows][:64])                 and 'nuisance' in z:
            return {k:z[k] for k in HQ_KEYS},z['nuisance']
    feats={k:[] for k in HQ_KEYS};nuis=[];meta=[]
    for r in bank_rows:
        f=hq.extract(hq.load_gray(r['image_path']),inverted=r.get('is_inverted') in ('1','True','true'))
        for k in HQ_KEYS:feats[k].append(f[k])
        nuis.append(f['nuisance']);meta.append(f['meta'])
    arrs={k:np.asarray(v,dtype=np.float32) for k,v in feats.items()}
    arrs['nuisance']=np.asarray(nuis,dtype=np.float32)
    arrs['version']=np.array(FEATURE_VERSION)
    arrs['rows_sha']=np.array(json.dumps([r.get('pixel_sha256') for r in bank_rows][:64]))
    cache.parent.mkdir(parents=True,exist_ok=True)
    np.savez(cache,**arrs)
    return arrs,arrs['nuisance']


def concat_hq(parts):
    return np.concatenate([parts['hinge_s'],parts['hinge_l'],parts['quill']],axis=1)


def real_queries(bank_rows,per_slot=48):
    bank_ids={r['img_id'] for r in bank_rows}
    by_slot=defaultdict(list)
    for r in load_real_rows():
        if r['img_id'] in bank_ids:continue
        if (r.get('aug') or '').strip():continue  # bank is unaugmented; queries match
        by_slot[(r['calligrapher'],r['script'])].append(r)
    queries=[]
    for slot,rows in sorted(by_slot.items()):
        rows=sorted(rows,key=lambda r:r['img_id'])
        queries+=rows[:per_slot]
    return queries


def throughput(paths,limit=500):
    paths=paths[:limit]
    t0=time.monotonic();meta=[]
    for pth in paths:
        f=hq.extract(hq.load_gray(pth))
        meta.append(f['meta'])
    dt=time.monotonic()-t0
    return dict(n=len(paths),seconds=round(dt,1),images_per_s=round(len(paths)/dt,2),
        extrapolate_20k_min=round(20000/len(paths)*dt/60,1),
        s_short_hist=sorted({m['s_short'] for m in meta})[:8])


def panel_query_feats(panel,cache_root=None,machine='48'):
    cache_root=cache_root or f'assets/eval_style_v2_cache/{machine}'
    rows=list(csv.DictReader(open(panel['csv'],encoding='utf-8-sig')))
    folder=Path(panel['png_root'])
    steps=sorted(folder.glob('eval_samples_ctrl/step*'))
    if not steps:return None
    pngs=[steps[0]/'eval200fix'/f'g{i}.png' for i in range(len(rows))]
    pngs=[p for p in pngs if p.is_file()]
    if len(pngs)<len(rows)*0.9:return None
    rows=rows[:len(pngs)]
    tag=f"{panel['arm']}_{panel['step']}_{panel['protocol']}"
    eg_cache=Path(cache_root)/'panels'/f'{tag}_eg.npz'
    eg_cache.parent.mkdir(parents=True,exist_ok=True)
    if eg_cache.is_file():
        q_eg=np.load(eg_cache)['E-G']
    else:
        q_eg=eg.eg_features([str(p) for p in pngs],cache_path=eg_cache)
    hq_cache=Path(cache_root)/'panels'/f'{tag}_hq.npz'
    if hq_cache.is_file():
        z=np.load(hq_cache);q_hq={k:z[k] for k in HQ_KEYS}
    else:
        q_hq={k:[] for k in HQ_KEYS}
        for pth in pngs:
            f=hq.extract(hq.load_gray(pth))
            for k in HQ_KEYS:q_hq[k].append(f[k])
        q_hq={k:np.asarray(v,dtype=np.float32) for k,v in q_hq.items()}
        np.savez(hq_cache,**q_hq)
    return dict(tag=tag,rows=rows,pngs=[str(p) for p in pngs],
        feats={'E-G':q_eg,**q_hq,'hq_concat':concat_hq(q_hq)})


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',default='docs/1030/eval_style_v2')
    p.add_argument('--machine',default='48')
    p.add_argument('--throughput-only',action='store_true')
    p.add_argument('--panels',action='store_true')
    a=p.parse_args()
    out=Path(a.out)
    machine=a.machine
    bank_rows,eg_bank=eg.bank_eg(machine)
    print('bank rows',len(bank_rows),flush=True)
    tp=throughput([r['image_path'] for r in bank_rows])
    print('throughput',json.dumps(tp),flush=True)
    if a.throughput_only:
            save_json(out/f'throughput_{machine}.json',tp);return
    hqb,bank_nuis=bank_hq(bank_rows,machine)
    bank_feats={'E-G':eg_bank,'hinge_s':hqb['hinge_s'],'hinge_l':hqb['hinge_l'],
        'quill':hqb['quill'],'hq_concat':concat_hq(hqb)}
    queries=real_queries(bank_rows)
    print('real queries',len(queries),flush=True)
    q_eg_cache=Path(f'assets/eval_style_v2_cache/{machine}/real_queries_eg.npz')
    if q_eg_cache.is_file():
        q_eg=np.load(q_eg_cache)['E-G']
    else:
        q_eg=eg.eg_features([r['image_path'] for r in queries],cache_path=q_eg_cache)
    q_hq_cache=Path(f'assets/eval_style_v2_cache/{machine}/real_queries_hq.npz')
    if q_hq_cache.is_file() and 'nuisance' in np.load(q_hq_cache):
        z=np.load(q_hq_cache);q_hq={k:z[k] for k in HQ_KEYS};q_nuis=z['nuisance']
    else:
        q_hq={k:[] for k in HQ_KEYS};q_nuis=[]
        for r in queries:
            f=hq.extract(hq.load_gray(r['image_path']),inverted=r.get('is_inverted') in ('1','True','true'))
            for k in HQ_KEYS:q_hq[k].append(f[k])
            q_nuis.append(f['nuisance'])
        q_hq={k:np.asarray(v,dtype=np.float32) for k,v in q_hq.items()}
        np.savez(q_hq_cache,**q_hq,**{'nuisance':np.asarray(q_nuis,dtype=np.float32)})
    real_feats={'E-G':q_eg,'hinge_s':q_hq['hinge_s'],'hinge_l':q_hq['hinge_l'],
        'quill':q_hq['quill'],'hq_concat':concat_hq(q_hq)}
    validation={};csv_rows=[]
    for feat in FEATURES:
        entry={}
        for k in ([32,64,128] if feat!='E-G' else [64]):
            res,_=M.evaluate_feature(real_feats[feat],queries,bank_feats[feat],bank_rows,k)
            entry[f'K{k}']=res
        perm,_=M.evaluate_feature(real_feats[feat],queries,bank_feats[feat],bank_rows,64,permutation_seed=11)
        entry['permutation_K64']=perm
        validation[feat]=entry
        base=entry.get('K64',entry.get('K32',{}))
        csv_rows.append(dict(split='real',model='RealGT',feature=feat,k=base.get('k'),
            n=base.get('n_valid'),auc=base.get('auc'),eer=base.get('eer'),
            map_macro=base.get('map_macro'),auc_ci95=json.dumps(base.get('auc_ci95'))))
        print('real',feat,json.dumps(entry),flush=True)
    save_json(out/'real_encoder_validation.json',dict(
        feature_version=FEATURE_VERSION,preprocess=PREPROCESS,
        bank=eg.eg_provenance(),n_queries=len(queries),throughput=tp,results=validation))
    np.save(out/f'nuisance_queries_{machine}.npy',q_nuis)
    np.save(out/f'nuisance_bank_{machine}.npy',bank_nuis)
    if a.panels:
        audit=read_json(out/f'data_audit_{machine}.json' if Path(out/f'data_audit_{machine}.json').is_file() else out/'data_audit.json')
        skipped=0
        for panel in audit['panels']:
            if not panel.get('csv'):skipped+=1;continue
            qp=panel_query_feats(panel,machine=machine)
            if qp is None:continue
            for feat in FEATURES:
                for k in ([32,64,128] if feat!='E-G' else [64]):
                    res,_=M.evaluate_feature(qp['feats'][feat],qp['rows'],bank_feats[feat],bank_rows,k)
                    csv_rows.append(dict(split='panel',model=f"{qp['tag']}",feature=feat,k=k,
                        n=res.get('n_valid'),auc=res.get('auc'),eer=res.get('eer'),
                        map_macro=res.get('map_macro'),auc_ci95=json.dumps(res.get('auc_ci95'))))
            print('panel',qp['tag'],flush=True)
        print('panels skipped (no csv identity):',skipped,flush=True)
    with open(out/'writer_verification.csv','w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['split','model','feature','k','n','auc','eer','map_macro','auc_ci95'])
        writer.writeheader();writer.writerows(csv_rows)
    print('WROTE',out/'writer_verification.csv',flush=True)


if __name__=='__main__':main()
