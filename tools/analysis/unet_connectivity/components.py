"""Threshold/area sensitivity is a diagnostic, never a postprocessing quality claim."""
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import label


def retained_components(mask,min_area):
    ids,count=label(mask,structure=np.ones((3,3),dtype=int))
    areas=np.bincount(ids.ravel())
    keep=areas>=min_area
    keep[0]=False
    filtered=keep[ids]
    return filtered,label(filtered,structure=np.ones((3,3),dtype=int))


def topology(pred,gt,threshold=.5,min_area=0):
    p,(pl,pn)=retained_components(pred<threshold,min_area)
    g,(gl,gn)=retained_components(gt<threshold,min_area)
    missing=sum((gl==i).sum()>=5 and p[gl==i].mean()<.25 for i in range(1,gn+1))
    merged=0
    for i in range(1,pn+1):
        ids,counts=np.unique(gl[pl==i],return_counts=True)
        merged+=int(((ids>0)&(counts>=5)).sum()>1)
    return dict(frag=pn/gn if gn else (1. if pn==0 else float(pn)),pred_components=pn,gt_components=gn,
                gt_empty=float(gn==0),ink_iou=float((p&g).sum()/max(1,(p|g).sum())),
                missing_components=int(missing),merged_components=int(merged),
                pred_mass=float(p.mean()),gt_mass=float(g.mean()))


def scan(directory,n=187,thresholds=(.3,.4,.5,.6,.7),areas=(0,2,4,8)):
    if n<1 or any(not 0<t<1 for t in thresholds) or any(a<0 for a in areas):
        raise ValueError('Positive sample count, thresholds in (0,1), nonnegative component areas required')
    directory=Path(directory)
    pairs=[(np.asarray(Image.open(directory/f'g{i}.png').convert('RGB')).mean(2)/255,
            np.asarray(Image.open(directory/f'gt{i}.png').convert('RGB')).mean(2)/255) for i in range(n)]
    output=[]
    for threshold in thresholds:
        for area in areas:
            values=[topology(p,g,threshold,area) for p,g in pairs]
            output.append(dict(threshold=threshold,min_area=area,n=n,
                               mean={k:float(np.mean([r[k] for r in values])) for k in values[0]},
                               per_image=values))
    return dict(directory=str(directory),connectivity=8,filter_applied_to='pred_and_gt',
                note='Original PNGs untouched; threshold/area-filtered diagnostics are not final quality scores',cells=output)
