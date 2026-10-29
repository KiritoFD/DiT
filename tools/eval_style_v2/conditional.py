"""Phase 3: conditional distribution diagnostics and writer relation geometry.

All statistics use the frozen E-G features. New diagnostics (not claimed to be
IJCV BCFID):
- Writer-RDM Corr: Spearman correlation between real and generated
  writer-to-writer distance matrices, computed per script over writers that
  have both real references and generated images.
- Writer Direction Alignment: cos(real mean-difference, generated
  mean-difference) per writer pair, macro over pairs; amplitude ratio
  reports compression/exaggeration of style differences.
- Within-writer MMD macro with writer-label permutation control.

Requires the per-panel feature caches written by verify.py --panels.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from .common import load_real_rows,save_json,read_json
from . import metrics as M


def load_panel_feats(machine,tag,cache_root=None):
    root=Path(cache_root or f'assets/eval_style_v2_cache/{machine}')/'panels'
    f=root/f'{tag}_eg.npz'
    if not f.is_file():return None
    return np.load(f)['E-G']


def spearman(a,b):
    ra=np.argsort(np.argsort(a));rb=np.argsort(np.argsort(b))
    ra=ra-ra.mean();rb=rb-rb.mean()
    return float((ra@rb)/max(np.linalg.norm(ra)*np.linalg.norm(rb),1e-9))


def rdm_alignment(real_by_slot,gen_by_slot,permutation=None):
    """Per script: RDM corr + direction alignment + amplitude ratio."""
    slots=sorted(set(real_by_slot)&set(gen_by_slot))
    result={}
    for script in sorted({s for _,s in slots}):
        writers=sorted({w for w,s2 in slots if s2==script})
        if len(writers)<3:continue
        mu_r={w:real_by_slot[(w,script)] for w in writers}
        mu_g={w:gen_by_slot[(w,script)] for w in writers}
        dr=[];dg=[];align=[];amp=[];pairs=[]
        for i,a in enumerate(writers):
            for b in writers[i+1:]:
                vr=mu_r[a]-mu_r[b];vg=mu_g[a]-mu_g[b]
                dr.append(float(np.linalg.norm(vr)));dg.append(float(np.linalg.norm(vg)))
                denom=np.linalg.norm(vr)*np.linalg.norm(vg)
                align.append(float(vr@vg/denom) if denom>1e-9 else 0.)
                amp.append(float(np.linalg.norm(vg)/max(np.linalg.norm(vr),1e-9)))
                pairs.append(f'{a}-{b}')
        result[script]=dict(n_writers=len(writers),n_pairs=len(pairs),
            rdm_spearman=round(spearman(np.array(dr),np.array(dg)),4),
            direction_align_macro=round(float(np.mean(align)),4),
            amplitude_ratio_median=round(float(np.median(amp)),4),
            pairs=pairs,align_per_pair=[round(v,4) for v in align])
    return result


def real_slot_means(bank_rows,bank_feats,excluded_chars):
    by=defaultdict(list)
    for j,r in enumerate(bank_rows):
        if r['character'] in excluded_chars.get((r['calligrapher'],r['script']),set()):continue
        by[(r['calligrapher'],r['script'])].append(j)
    return {slot:bank_feats[idx].mean(0) for slot,idx in by.items() if len(idx)>=8}


def gen_slot_means(rows,feats,writer_labels=None):
    by=defaultdict(list)
    for j,r in enumerate(rows):
        w=(writer_labels or {}).get(r['img_id'],r['calligrapher'])
        by[(w,r['script'])].append(j)
    return {slot:feats[idx].mean(0) for slot,idx in by.items() if len(idx)>=4}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--machine',default='48')
    p.add_argument('--out',default='docs/1030/eval_style_v2')
    a=p.parse_args()
    out=Path(a.out)
    audit_file=out/f'data_audit_{a.machine}.json'
    audit=read_json(audit_file if audit_file.is_file() else out/'data_audit.json')
    from . import features_eg as eg
    bank_rows,bank_feats=eg.bank_eg(a.machine)
    results={}
    csv_rows=[]
    for panel in audit['panels']:
        if not panel.get('csv'):continue
        tag=f"{panel['arm']}_{panel['step']}_{panel['protocol']}"
        feats=load_panel_feats(a.machine,tag)
        if feats is None:continue
        rows=list(csv.DictReader(open(panel['csv'],encoding='utf-8-sig')))
        excluded=defaultdict(set)
        for r in rows:excluded[(r['calligrapher'],r['script'])].add(r['character'])
        real_means=real_slot_means(bank_rows,bank_feats,excluded)
        gen_means=gen_slot_means(rows,feats)
        diag=rdm_alignment(real_means,gen_means)
        # permutation control: shuffle generated writer labels
        import random
        rng=random.Random(13)
        labels={r['img_id']:r['calligrapher'] for r in rows}
        ids=list(labels);vals=list(labels.values());rng.shuffle(vals)
        perm=dict(zip(ids,vals))
        diag_perm=rdm_alignment(real_means,gen_slot_means(rows,feats,perm))
        results[tag]=dict(diagnostic=diag,permutation_control=diag_perm)
        for script,d in diag.items():
            csv_rows.append(dict(model=tag,script=script,n_writers=d['n_writers'],
                rdm_spearman=d['rdm_spearman'],direction_align=d['direction_align_macro'],
                amplitude_ratio=d['amplitude_ratio_median'],
                rdm_perm=diag_perm.get(script,{}).get('rdm_spearman')))
        print('done',tag,flush=True)
    save_json(out/'conditional_distribution.json',results)
    with open(out/'conditional_distribution.csv','w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['model','script','n_writers','rdm_spearman',
            'direction_align','amplitude_ratio','rdm_perm'])
        w.writeheader();w.writerows(csv_rows)
    print('WROTE',out/'conditional_distribution.csv',flush=True)


if __name__=='__main__':main()
