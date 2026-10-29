"""统一评测: per-arm metrics from saved panels/swap PNGs (CPU-friendly).

Inputs: swap manifests + val100 evaluation result.json (from io.evaluate runs
launched by the chain) + generated PNG dirs. Outputs matched_eval.csv and
writer_swap.csv with E-G / E-G2 retrieval, direction, amplitude, content
stability, near-blank and width diagnostics.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from .common import EGEncoder,EG2Encoder,save_json,PILOT_ROOT,TMP_ROOT
from tools.eval_unpaired.protocol import reference_bank
from tools.eval_style_v2 import metrics as M


def png_feats(encoder,paths,batch=32):
    from PIL import Image
    feats=[]
    for start in range(0,len(paths),batch):
        chunk=[np.asarray(Image.open(p).convert('RGB'),dtype=np.float32)/255. for p in paths[start:start+batch]]
        x=torch.from_numpy(np.stack(chunk)).permute(0,3,1,2).float()
        with torch.no_grad():feats.append(encoder.features(x).cpu().numpy())
    return np.concatenate(feats)


def swap_metrics(manifest,eg,eg2,bank_rows,bank_feats,eg2_bank):
    imgs=defaultdict(list)
    for m in manifest:imgs[(m['script'],m['seed'],m['char'])].append(m)
    dir_align=[];amp=[];stab=[];rows=[]
    real_mean=defaultdict(list)
    for j,r in enumerate(bank_rows):
        real_mean[(int(r['calligrapher_id']),r['script'])].append(bank_feats[j])
    real_mean={k:np.mean(v,axis=0) for k,v in real_mean.items() if len(v)>=8}
    by_slot_seed=defaultdict(lambda:defaultdict(list))
    for key,ms in imgs.items():
        for m in ms:by_slot_seed[(m['script'],m['seed'])][m['cond']].append(m)
    for (script,seed),conds in sorted(by_slot_seed.items()):
        writers=sorted({m['writer'] for m in conds['correct']})
        gen_mean=defaultdict(list)
        for m in conds['correct']:
            f=eg.features_numpy(__import__('tools.eg_pilot48.common',fromlist=['bank_tensors']).bank_tensors([m['path']]))[0]
            gen_mean[m['writer']].append(f)
        gen_mean={w:np.mean(v,axis=0) for w,v in gen_mean.items() if len(v)>=3}
        for i,wa in enumerate(writers):
            for wb in writers[i+1:]:
                if wa not in gen_mean or wb not in gen_mean:continue
                if (wa,script) not in real_mean or (wb,script) not in real_mean:continue
                dr=real_mean[(wa,script)]-real_mean[(wb,script)]
                dg=gen_mean[wa]-gen_mean[wb]
                denom=np.linalg.norm(dr)*np.linalg.norm(dg)
                align=float(dr@dg/denom) if denom>1e-9 else 0.
                dir_align.append(align);amp.append(float(np.linalg.norm(dg)/max(np.linalg.norm(dr),1e-9)))
        # content stability: correct vs null / wrong per char+seed
        for (script2,seed2,ch),ms in imgs.items():
            if seed2!=seed or script2!=script:continue
            cm=[m for m in ms if m['cond']=='correct'];nm=[m for m in ms if m['cond']=='null']
            wm=[m for m in ms if m['cond']=='wrong']
            if cm and nm:
                from skimage.metrics import structural_similarity as ssim_fn
                from PIL import Image
                a=np.asarray(Image.open(cm[0]['path']).convert('L'),dtype=np.float32)/255.
                b=np.asarray(Image.open(nm[0]['path']).convert('L'),dtype=np.float32)/255.
                stab.append(dict(script=script,char=ch,seed=seed,kind='null',
                    ssim=round(float(ssim_fn(a,b,data_range=1.)),4)))
            if cm and wm:
                from skimage.metrics import structural_similarity as ssim_fn
                from PIL import Image
                a=np.asarray(Image.open(cm[0]['path']).convert('L'),dtype=np.float32)/255.
                b=np.asarray(Image.open(wm[0]['path']).convert('L'),dtype=np.float32)/255.
                stab.append(dict(script=script,char=ch,seed=seed,kind='wrong',
                    ssim=round(float(ssim_fn(a,b,data_range=1.)),4)))
    summary=dict(direction_align_macro=round(float(np.mean(dir_align)),4) if dir_align else None,
        n_pairs=len(dir_align),
        amplitude_median=round(float(np.median(amp)),4) if amp else None,
        content_stability_correct_vs_null=round(float(np.mean([s['ssim'] for s in stab if s['kind']=='null'])),4) if any(s['kind']=='null' for s in stab) else None,
        content_stability_correct_vs_wrong=round(float(np.mean([s['ssim'] for s in stab if s['kind']=='wrong'])),4) if any(s['kind']=='wrong' for s in stab) else None)
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--arms',nargs='+',required=True)
    p.add_argument('--out',default=PILOT_ROOT)
    a=p.parse_args()
    out=Path(a.out)
    eg=EGEncoder()
    try:eg2=EG2Encoder()
    except Exception:eg2=None
    bank_rows=reference_bank(REAL_CSV)
    from tools.eg_pilot48.common import bank_tensors as BT
    bank_feats=eg.features_numpy(BT([r['image_path'] for r in bank_rows]).to('cuda'))
    eg2_bank=None
    if eg2 is not None:eg2_bank=eg2.features_numpy(BT([r['image_path'] for r in bank_rows]).to('cuda'))
    all_rows=[]
    for tag in a.arms:
        swap_dir=TMP_ROOT/f'swap_{tag}' if tag!='S5_START' else TMP_ROOT/'swap_S5_START'
        mf=swap_dir/'manifest.json'
        if not mf.is_file():print('no swap for',tag);continue
        manifest=json.loads(mf.read_text())
        feats=png_feats(eg,[m['path'] for m in manifest])
        feats2=png_feats(eg2,[m['path'] for m in manifest]) if eg2 is not None else None
        for i,m in enumerate(manifest):m['_eg']=feats[i];m['_eg2']=feats2[i] if feats2 is not None else None
        # E-G retrieval per condition (correct-writer images identified as their writer)
        for cond in ('correct',):
            q=[m for m in manifest if m['cond']==cond]
            if not q:continue
            qf=np.stack([m['_eg'] for m in q])
            qrows=[dict(calligrapher=str(m['writer']),script=str(m['script']),character=m['char'],
                pixel_sha256=None,img_id=f'{m["writer"]}_{m["char"]}_{m["seed"]}') for m in q]
            res,_=M.evaluate_feature(qf,qrows,bank_feats,bank_rows,64)
            all_rows.append(dict(model=tag,metric='eg_auc',cond=cond,value=res.get('auc'),n=res.get('n_valid')))
            if eg2 is not None:
                qf2=np.stack([m['_eg2'] for m in q])
                res2,_=M.evaluate_feature(qf2,qrows,eg2_bank,bank_rows,64)
                all_rows.append(dict(model=tag,metric='eg2_auc',cond=cond,value=res2.get('auc'),n=res2.get('n_valid')))
        summary=swap_metrics(manifest,eg,eg2,bank_rows,bank_feats,eg2_bank)
        for k,v in summary.items():all_rows.append(dict(model=tag,metric=k,cond='',value=v,n=''))
        print(tag,json.dumps(summary),flush=True)
    with open(out/'writer_swap.csv','w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['model','metric','cond','value','n'])
        w.writeheader();w.writerows(all_rows)
    print('WROTE',out/'writer_swap.csv',flush=True)


if __name__=='__main__':main()
