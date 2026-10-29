"""P125 post-hoc checkpoint evaluation: the frozen metric table.

Given a run dir with a completed in-queue heldout eval (val100 panel PNGs),
computes on CPU: E-G verification (AUC/EER/mAP/2AFC vs the Top10 real bank),
LPIPS-VGG, SSIM, Ink-IoU, Near-Blank, SoftInk/Width ratio. Writes one JSON
plus one row for p125_metrics.csv. E-G scores of P3-B runs are diagnostics.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

from tools.eval_style_v2 import features_eg as eg
from tools.eval_style_v2 import metrics as M

EG_MEAN=torch.tensor([.485,.456,.406]);EG_STD=torch.tensor([.229,.224,.225])


def eg_embed_paths(paths,net,size=160):
    from PIL import Image
    out=[]
    with torch.no_grad():
        for i in range(0,len(paths),64):
            batch=torch.from_numpy(np.stack([np.asarray(
                Image.open(p).convert('RGB').resize((256,256),Image.Resampling.BICUBIC)).copy()
                for p in paths[i:i+64]])).permute(0,3,1,2).float()/255
            x=F.interpolate(batch,(size,size),mode='bilinear',align_corners=False)
            x=(x-EG_MEAN.to(x.device)[None,:,None,None])/EG_STD.to(x.device)[None,:,None,None]
            e=net.emb(x)
            out.append((e/e.norm(dim=-1,keepdim=True).clamp_min(1e-6)).cpu())
    return torch.cat(out)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',required=True)
    p.add_argument('--step',type=int,required=True)
    p.add_argument('--protocol',default='heldout')
    p.add_argument('--csv',default='assets/u100_pretrain_20261024/validation_val100.csv')
    p.add_argument('--arm',default='')
    p.add_argument('--out',default='docs/1030/eval_style_v2/p125_metrics.csv')
    a=p.parse_args()
    run=Path(a.run)
    folder=run/'evaluations'/f'{a.step:07d}'/a.protocol
    steps=sorted(folder.glob('eval_samples_ctrl/step*'))
    pngs=[steps[0]/'eval200fix'/f'g{i}.png' for i in range(200)]
    pngs=[p for p in pngs if p.is_file()]
    rows=list(csv.DictReader(open(a.csv,encoding='utf-8-sig')))[:len(pngs)]
    net,kind,_=__import__('tools.eval_gen_with_disc',fromlist=['x']).load_evaluator_model(
        'assets/style_disc_ce_holdout.pt','cpu')
    q=eg_embed_paths([str(p) for p in pngs],net).numpy()
    bank_rows,bank_feats=eg.bank_eg('4090')
    res=M.evaluate_feature(q,rows,bank_feats,bank_rows,64)
    # 2AFC = mean afc
    dec=M.collect_decisions(q,rows,bank_feats,bank_rows,64)
    afc=[d['scores'] and None for d in []]  # placeholder replaced below
    afcs=[]
    for d in dec:
        t=d['target']
        if t not in d['scores']:continue
        others=[v for w,v in d['scores'].items() if w!=t]
        if not others:continue
        afcs.append(float(np.mean([d['scores'][t]>v for v in others])))
    afc=float(np.mean(afcs)) if afcs else None
    # content: LPIPS-VGG, SSIM, Ink-IoU, near-blank, width ratio
    import lpips as lpips_mod
    from skimage.metrics import structural_similarity as ssim_fn
    from PIL import Image
    loss_vgg=lpips_mod.LPIPS(net='vgg',verbose=False)
    gt_paths={r['img_id']:r['image_path'] for r in rows}
    ssims=[];lpipss=[];ious=[];blanks=[];inks=[];wr=[]
    with torch.no_grad():
        for i,(png,r) in enumerate(zip(pngs,rows)):
            gen=np.asarray(Image.open(png).convert('L'),dtype=np.float32)/255.
            gt=np.asarray(Image.open(r['image_path']).convert('L'),dtype=np.float32)/255.
            ssims.append(ssim_fn(gt,gen,data_range=1.))
            t0=torch.from_numpy(np.stack([gt]*3))[None].float()*2-1
            t1=torch.from_numpy(np.stack([gen]*3))[None].float()*2-1
            lpipss.append(float(loss_vgg(t0,t1).item()))
            gmi=gen<.5;gti=gt<.5
            ious.append(float((gmi&gti).sum()/max((gmi|gti).sum(),1)))
            ink=float(gmi.mean())
            inks.append(ink)
            if ink<0.005:blanks.append(1)
            er=__import__('scipy.ndimage',fromlist=['ndimage']).binary_erosion(gmi,np.ones((3,3),bool))
            import scipy.ndimage as ndi
            dt=ndi.distance_transform_edt(gmi)
            wr.append(float(np.median(dt[gmi])*2/8.) if gmi.any() else 0.)
    record=dict(arm=a.arm or run.name,step=a.step,protocol=a.protocol,n=len(pngs),
        eg_auc=res.get('auc'),eg_eer=res.get('eer'),eg_map=res.get('map_macro'),
        eg_2afc=round(100*afc,2) if afc else None,
        ssim=round(float(np.mean(ssims)),4),lpips_vgg=round(float(np.mean(lpipss)),4),
        ink_iou=round(float(np.mean(ious)),4),near_blank_rate=round(sum(blanks)/len(pngs),4),
        soft_ink_ratio=round(float(np.mean(inks)),4),width_ratio_median=round(float(np.mean(wr)),4),
        eg_note='diagnostic-only for P3B runs' if 'P3B' in a.arm or 'P3B' in run.name else '')
    print(json.dumps(record,ensure_ascii=False))
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    exists=out.is_file()
    with open(out,'a',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(record.keys()))
        if not exists:w.writeheader()
        w.writerow(record)


if __name__=='__main__':main()
