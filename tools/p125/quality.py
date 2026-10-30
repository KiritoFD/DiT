"""Unsupervised visual-quality metrics (plan §5.1) + real-calibrated anomaly rates.

Defect features per image (binary ink mask, no GT needed):
  ink_ratio, n_components, fragment_share (small components' ink share),
  width_p50/p90 (distance transform), width_jump (skeleton width variation),
  edge_roughness (boundary gradient energy), endpoint_density.
All are morphology proxies, NOT aesthetic scores. Anomaly = feature outside the
Real-vs-Real reference quantiles (frozen from a real calibration pass).
"""
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from PIL import Image


def defect_features(gray):
    mask=gray<0.5
    if mask.mean()>0.5:mask=~mask
    ink=float(mask.mean())
    if ink<0.002:
        return dict(ink_ratio=ink,n_components=0,fragment_share=1.,width_p50=0.,
            width_p90=0.,width_jump=0.,edge_roughness=0.,endpoint_density=0.)
    lab,n=ndi.label(mask,structure=np.ones((3,3),int))
    sizes=ndi.sum(mask,lab,range(1,n+1)) if n else np.array([])
    total=mask.sum()
    frag_share=float(sizes[sizes<max(total*0.005,4)].sum()/total) if n else 0.
    dt=ndi.distance_transform_edt(mask)
    width_map=2*dt
    width=width_map[mask]
    w50=float(np.median(width));w90=float(np.percentile(width,90))
    sk=mask&(dt>=np.maximum(dt-1,0.5))  # ridge approx of skeleton
    skw=width_map[sk] if sk.any() else np.array([0.])
    gy,gx=np.gradient(dt)
    gm=np.hypot(gx,gy)
    er=float(gm[~mask].mean()) if (~mask).any() else 0.
    erood=ndi.binary_erosion(mask,np.ones((3,3),bool))
    boundary=mask&~erood
    be=gm[boundary] if boundary.any() else np.array([0.])
    neigh=ndi.convolve(sk.astype(float),np.ones((3,3)),mode='constant')
    ends=sk&(neigh<=2.0)
    return dict(ink_ratio=ink,n_components=int(n),
        fragment_share=round(frag_share,4),
        width_p50=round(w50,3),width_p90=round(w90,3),
        width_jump=round(float(skw.std()/max(skw.mean(),1e-6)),4),
        edge_roughness=round(float(er+be.mean()*0.5),4),
        endpoint_density=round(float(ends.sum()/max(total,1)),6))


def load_gray(path):
    return np.asarray(Image.open(path).convert('L'),dtype=np.float32)/255.


FEATS=('ink_ratio','n_components','fragment_share','width_p50','width_p90',
       'width_jump','edge_roughness','endpoint_density')


def calibrate_real(image_paths,out):
    feats=[defect_features(load_gray(p)) for p in image_paths]
    calib={}
    for k in FEATS:
        v=np.array([f[k] for f in feats],dtype=float)
        calib[k]=dict(p05=round(float(np.percentile(v,5)),6),
            p50=round(float(np.percentile(v,50)),6),p95=round(float(np.percentile(v,95)),6))
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    Path(out).write_text(json.dumps(dict(n=len(feats),quantiles=calib),indent=1))
    return calib


def anomaly_rates(panel_paths,calib):
    per_feat=defaultdict(list)
    for p in panel_paths:
        f=defect_features(load_gray(p))
        for k in FEATS:per_feat[k].append(f[k])
    rates={}
    for k in FEATS:
        v=np.array(per_feat[k],dtype=float)
        lo,hi=calib[k]['p05'],calib[k]['p95']
        rates[k]=dict(median=round(float(np.median(v)),6),
            anomaly_rate=round(float(np.mean((v<lo)|(v>hi))),4))
    overall=float(np.mean([rates[k]['anomaly_rate'] for k in FEATS]))
    rates['overall_anomaly_rate']=round(overall,4)
    return rates
