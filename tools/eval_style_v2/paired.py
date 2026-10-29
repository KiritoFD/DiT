"""Phase 5: paired per-character comparison S5-S0 / S5-S4 / S4-S0 @50k panels.

Style evidence = E-G retrieval margin per character (frozen bank, K64).
Content evidence = per-image SSIM / LPIPS / Ink-IoU recomputed on CPU against
the panel's GT rows. Pre-registered tolerance: content non-regression means
dSSIM >= -0.01 (frozen here before looking at S5 results).

A/B/C/D classes: A style+ / content-ok, B style+ / content-regress,
C style- / content-ok, D style- / content-regress.
Panels may live on different machines: S5 panel features are copied in as
panel_s5 (npz) + PNG dir; S0/S4 are local.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from .common import load_real_rows,save_json,read_json
from . import features_eg as eg
from . import metrics as M

TOL_SSIM=-0.01


def panel_rows_feats(machine,arm,protocol,out='docs/1030/eval_style_v2',png_root_override=None):
    audit_path=Path(out)/f'data_audit_{machine}.json'
    audit=read_json(audit_path if audit_path.is_file() else Path(out)/'data_audit.json')
    panel=next(p for p in audit['panels'] if p['arm']==arm and p['protocol']==protocol and p.get('csv'))
    if png_root_override:panel=dict(panel,png_root=png_root_override)
    rows=list(csv.DictReader(open(panel['csv'],encoding='utf-8-sig')))
    cache=Path(f'assets/eval_style_v2_cache/{machine}')/'panels'/f"{arm}_{panel['step']}_{protocol}_eg.npz"
    feats=np.load(cache)['E-G']
    return rows,feats,panel


def content_metrics(png_root,rows):
    """Per-image SSIM/LPIPS/Ink-IoU against the CSV GT rows (CPU)."""
    import torch
    torch.set_num_threads(6)
    import lpips as lpips_mod
    from skimage.metrics import structural_similarity as ssim_fn
    loss_alex=lpips_mod.LPIPS(net='alex',verbose=False)
    steps=sorted(Path(png_root).glob('eval_samples_ctrl/step*'))
    pngs={i:steps[0]/'eval200fix'/f'g{i}.png' for i in range(len(rows))}
    result={}
    for i,row in enumerate(rows):
        gp=pngs.get(i)
        if gp is None or not gp.is_file():continue
        gt_row=row
        if not gt_row.get('image_path'):continue
        from PIL import Image
        gen=np.asarray(Image.open(gp).convert('L'),dtype=np.float32)/255.
        gt=np.asarray(Image.open(gt_row['image_path']).convert('L'),dtype=np.float32)/255.
        s=ssim_fn(gt,gen,data_range=1.0)
        with torch.no_grad():
            t0=torch.from_numpy(np.stack([gen]*3))[None].float()*2-1
            t1=torch.from_numpy(np.stack([gt]*3))[None].float()*2-1
            lp=float(loss_alex(t1,t0).item())
        gmi=gen<0.5;gti=gt<0.5
        inter=(gmi&gti).sum();union=(gmi|gti).sum()
        iou=float(inter/max(union,1))
        result[i]=dict(ssim=round(float(s),4),lpips=round(lp,4),ink_iou=round(iou,4),
            character=row['character'],writer=row['calligrapher'])
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',default='docs/1030/eval_style_v2')
    p.add_argument('--skip-content',action='store_true')
    a=p.parse_args()
    out=Path(a.out)
    bank_rows,bank_feats=eg.bank_eg('4090')
    panels={}
    config=(('S0','4090','heldout',None),('S4','4090','heldout',None),
            ('S5','48','heldout_s','assets/style_matrix_20261027/matrix64/48_S5_staging'))
    for arm,machine,protocol,override in config:
        panels[arm]=panel_rows_feats(machine,arm,protocol,out,png_root_override=override)
    margins={};chars=None
    contents={}
    for arm,(rows,feats,panel) in panels.items():
        m,ch=M_collect(feats,rows,bank_rows,bank_feats)
        margins[arm]=m;chars=ch
        if not a.skip_content:
            contents[arm]=content_metrics(panel['png_root'],rows)
            print('DBG contents',arm,len(contents[arm] or {}),flush=True)
    deltas={}
    csv_rows=[]
    for hi,lo in (('S5','S0'),('S5','S4'),('S4','S0')):
        common=sorted(set(margins[hi])&set(margins[lo]))
        recs=[]
        for key in common:
            w,ch,sc=chars[key]
            dsty=margins[hi][key]-margins[lo][key]
            rec=dict(idx=key,writer=w,character=ch,script=sc,d_style=round(dsty,4))
            if not a.skip_content and hi in contents and lo in contents \
                    and key in contents[hi] and key in contents[lo]:
                c_hi=contents[hi][key];c_lo=contents[lo][key]
                rec.update(d_ssim=round(c_hi['ssim']-c_lo['ssim'],4),
                    d_lpips=round(c_hi['lpips']-c_lo['lpips'],4),
                    d_ink_iou=round(c_hi['ink_iou']-c_lo['ink_iou'],4))
            recs.append(rec)
        deltas[f'{hi}-{lo}']=recs
    # A/B/C/D classification on the style-primary comparison S5-S0
    classes=defaultdict(list)
    for r in deltas.get('S5-S0',[]):
        if 'd_ssim' not in r:continue
        style_up=r['d_style']>0;content_ok=r['d_ssim']>=TOL_SSIM
        cls='ABCD'[(0 if style_up else 2)+(0 if content_ok else 1)]
        classes[cls].append(r['character'])
    summary={k:dict(n=len(v),share=round(len(v)/max(sum(len(x) for x in classes.values()),1),3)) for k,v in classes.items()}
    per_writer=defaultdict(lambda:defaultdict(list))
    for name,recs in deltas.items():
        for r in recs:
            per_writer[name][r['writer']].append(r['d_style'])
    attribution={name:{w:round(float(np.mean(v)),4) for w,v in sorted(m.items())} for name,m in per_writer.items()}
    save_json(out/'paired_comparison.json',dict(tolerance_dssim=TOL_SSIM,summary=summary,
        attribution_per_writer=attribution,deltas=deltas))
    with open(out/'paired_comparison.csv','w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['pair','idx','writer','character','script','d_style','d_ssim','d_lpips','d_ink_iou'])
        w.writeheader()
        for name,recs in deltas.items():
            for r in recs:w.writerow(dict(pair=name,**r))
    with open(out/'per_writer_report.csv','w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['pair','writer','mean_d_style','n'])
        w.writeheader()
        for name,m in attribution.items():
            for wr,v in m.items():
                w.writerow(dict(pair=name,writer=wr,mean_d_style=v,n=len(per_writer[name][wr])))
    print(json.dumps(dict(classes=summary,
        mean_d_style={k:round(float(np.mean([r['d_style'] for r in v])),4) for k,v in deltas.items()}),ensure_ascii=False))


def M_collect(feats,rows,bank_rows,bank_feats):
    dec=M.collect_decisions(feats,rows,bank_feats,bank_rows,64)
    margins={};chars={}
    for d in dec:
        target=d['target']
        if target not in d['scores']:continue
        others=[v for w,v in d['scores'].items() if w!=target]
        if not others:continue
        key=d['idx']
        margins[key]=d['scores'][target]-max(others)
        chars[key]=(d['row']['calligrapher'],d['row']['character'],d['row']['script'])
    return margins,chars


if __name__=='__main__':main()
