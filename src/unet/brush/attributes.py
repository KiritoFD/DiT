"""Deterministic five-dimensional brush measurements, ink=1 / background=0."""
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

NAMES=('width_p20','width_p50','width_p80','ink_density','edge_softness')


def measure(ink):
    ink=np.clip(np.asarray(ink,dtype=np.float32),0,1);mask=ink>=.5
    y,x=np.where(mask)
    if len(y)<32:return np.zeros(5,np.float32),False
    height=int(y.max()-y.min()+1)
    if height<8:return np.zeros(5,np.float32),False
    skel=skeletonize(mask);neighbors=ndi.convolve(skel.astype(np.int16),np.ones((3,3),np.int16))-skel
    junction=ndi.binary_dilation(skel&(neighbors>2),iterations=2)
    center=skel&~junction&(neighbors==2)
    widths=2*ndi.distance_transform_edt(mask)[center]/height
    if len(widths)<16:return np.zeros(5,np.float32),False
    perimeter=mask&~ndi.binary_erosion(mask)
    # Area/perimeter estimates average gray transition thickness, not a learned edge descriptor.
    softness=((ink>.1)&(ink<.9)).sum()/max(int(perimeter.sum()),1)/height
    vector=np.r_[np.quantile(widths,[.2,.5,.8]),ink[mask].mean(),softness].astype(np.float32)
    valid=bool(np.isfinite(vector).all() and center.sum()/max(skel.sum(),1)>=.2)
    return vector,valid


def normalize(values,scripts,train,valid):
    centers=[];scales=[]
    for sid in range(3):
        x=values[(scripts==sid)&train&valid]
        if len(x)<100:raise ValueError('Not enough valid training attributes for script '+str(sid))
        median=np.median(x,axis=0);scale=1.4826*np.median(np.abs(x-median),axis=0)
        centers.append(median);scales.append(np.maximum(scale,[.002,.002,.002,.01,.0005]))
    center=np.asarray(centers);scale=np.asarray(scales)
    normalized=np.clip((values-center[scripts])/scale[scripts],-5,5).astype(np.float32)
    normalized[~valid]=0
    return normalized,center,scale
