"""Last hand-crafted pilot: conditional, width/angle-aligned stroke patches.

128px white-background gray; skeleton junctions/endpoints excluded. No global
direction frequencies, patch counts or group occupancy are descriptor inputs.
Groups: four axial directions x width (<8px / >=8px) x straight/bent (.20rad).
Distance must compare the SAME group intersection for anchor/positive/negative.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

VERSION='local-stroke-v1-p128-grid17-g16-min3'
FAMILIES={'ink_edge':20,'geometry':50,'bend_width':8}


def extract_local(gray):
    gray=np.clip(np.asarray(gray,float),0,1)
    assert gray.shape==(128,128)
    g=np.clip(gray/max(float(np.quantile(gray,.95)),.5),0,1)
    dark=1-g;mask=g<.55;sk=skeletonize(mask)
    degree=ndi.convolve(sk.astype(int),np.ones((3,3),int))-sk
    valid=sk&(degree==2)&~ndi.binary_dilation(sk&(degree>2),iterations=2)
    sy,sx=np.mgrid[-3:4,-3:4];skf=sk.astype(float)
    mass=ndi.convolve(skf,np.ones((7,7)));den=np.maximum(mass,1)
    ex=ndi.correlate(skf,sx)/den;ey=ndi.correlate(skf,sy)/den
    vx=ndi.correlate(skf,sx*sx)/den-ex*ex
    vy=ndi.correlate(skf,sy*sy)/den-ey*ey
    xy=ndi.correlate(skf,sx*sy)/den-ex*ey
    angle=.5*np.arctan2(2*xy,vx-vy)
    anis=np.sqrt((vx-vy)**2+4*xy**2)/np.maximum(vx+vy,1e-9)
    edt=ndi.distance_transform_edt(mask)
    yy,xx=np.nonzero(valid&(mass>=5)&(anis>.35)&(edt>=1))
    counts=np.zeros(16,int)
    result={k:np.full((16,v),np.nan,np.float32) for k,v in FAMILIES.items()}
    if len(yy)<8:return {**result,'counts':counts}
    theta=angle[yy,xx];radius=edt[yy,xx];width=2*radius
    # Neighborhood tangent rotation measured with doubled-angle vectors, so
    # the 0/pi boundary cannot create fake 180deg curvature.
    tx=np.cos(theta);ty=np.sin(theta)
    points=[np.array([yy+sign*ty*radius,xx+sign*tx*radius]) for sign in (-1,1)]
    local=[]
    for coords in points:
        cc=ndi.map_coordinates(np.cos(2*angle),coords,order=1,mode='nearest')
        ss=ndi.map_coordinates(np.sin(2*angle),coords,order=1,mode='nearest')
        local.append(.5*np.arctan2(ss,cc))
    bend=np.abs(np.angle(np.exp(2j*(local[1]-local[0]))))/2
    orientation=np.floor(((theta+np.pi/8)%np.pi)/(np.pi/4)).astype(int)
    group=orientation*4+(width>=8).astype(int)*2+(bend>=.20).astype(int)
    u,v=np.meshgrid(np.linspace(-2,2,17),np.linspace(-1.5,1.5,17),indexing='ij')
    for gid in range(16):
        ids=np.flatnonzero(group==gid);counts[gid]=len(ids)
        if len(ids)<3:continue
        # Up to 32 regularly spaced samples per group, no embedding-based choice.
        ids=ids[np.linspace(0,len(ids)-1,min(len(ids),32)).astype(int)]
        ink=[];geometry=[];bw=[]
        for i in ids:
            patches=[]
            for scale in (1.,2.):
                y=yy[i]+radius[i]*scale*(u*ty[i]+v*tx[i])
                x=xx[i]+radius[i]*scale*(u*tx[i]-v*ty[i])
                patch=ndi.map_coordinates(dark,[y,x],order=1,mode='constant',cval=0)
                # PCA tangent sign is arbitrary; enforce both axial reflections.
                patch=(patch+patch[::-1]+patch[:,::-1]+patch[::-1,::-1])/4
                patches.append(patch)
            patch=patches[0];profile=patch[6:11].mean(0)[::2]
            across=patch[6:11].std(0)[::2]
            ink.append(np.r_[profile,across,patch[6:11,6:11].mean(),np.abs(np.diff(patch,axis=1)).mean()])
            geometry.append(np.r_[patches[0][::4,::4].ravel(),patches[1][::4,::4].ravel()])
            widths=(patch>.45).sum(1)/17
            bw.append([bend[i],widths.mean(),widths.std(),*np.quantile(widths,[.1,.5,.9]),
                       patch.std(),np.abs(np.diff(patch,axis=0)).mean()])
        for key,values in [('ink_edge',ink),('geometry',geometry),('bend_width',bw)]:
            result[key][gid]=np.mean(values,axis=0)
    return {**result,'counts':counts}


def triplet_margin(features,pairs,valid_groups):
    """Positive margin means d(anchor, positive) < d(anchor, negative)."""
    a,p,n=features[pairs[:,0]],features[pairs[:,1]],features[pairs[:,2]]
    shared=valid_groups[pairs].all(1)
    # Same groups and weights in both distances; no pair-dependent denominator.
    dp=np.nan_to_num(((a-p)**2).mean(-1),nan=0.)
    dn=np.nan_to_num(((a-n)**2).mean(-1),nan=0.)
    margin=((dn-dp)*shared).sum(1)/np.maximum(shared.sum(1),1)
    return margin,shared.sum(1)
