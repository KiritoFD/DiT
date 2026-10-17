"""Deterministic, non-learned glyph statistics. Angles are image-coordinate radians.

Quill-like and Hinge-like descriptors here are explicit custom approximations,
not claimed reproductions of published implementations. Fixed histogram edges
are shared across images. Invalid geometry stays NaN, never perfect zeros.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from skimage.measure import find_contours

VERSION = 'fingerprint-v1-128-th055-pca7'


def matched_pairs(query,train,w,s,c,source,stats):
    """Same pixel/metadata-only selection protocol as accept_teacher_matched.

    Kept dependency-free: this statistics job must not load a teacher network.
    """
    pairs=[];cross=0
    for i in query:
        if not np.isfinite(stats[i]).all():continue
        def match(indices):
            delta=np.abs(stats[indices]-stats[i])
            return indices[np.isfinite(delta).all(1)&(delta[:,0]<=.02)&(delta[:,1]<=.15*max(stats[i,1],1e-6))]
        p=match(train[(w[train]==w[i])&(s[train]==s[i])&(c[train]!=c[i])])
        preferred=p[source[p]!=source[i]];p=preferred if len(preferred) else p
        n=match(query[(w[query]!=w[i])&(s[query]==s[i])&(c[query]==c[i])])
        if not len(p) or not len(n):continue
        def nearest(indices):
            distance=(stats[indices,0]-stats[i,0])**2/.02**2+(stats[indices,1]-stats[i,1])**2/max(.15*stats[i,1],1e-6)**2
            return int(indices[np.argmin(distance)])
        j,k=nearest(p),nearest(n);pairs.append((int(i),j,k));cross+=source[j]!=source[i]
    return np.asarray(pairs,dtype=int).reshape(-1,3),int(cross)


def hist(x, edges):
    h = np.histogram(x, edges)[0].astype(float)
    return h/max(h.sum(), 1)


def spectrum(x):
    h,w = x.shape
    yy,xx = np.meshgrid(np.fft.fftfreq(h),np.fft.fftfreq(w), indexing='ij')
    r = np.hypot(xx,yy); a = np.arctan2(yy,xx)%np.pi
    window = np.outer(np.hanning(h),np.hanning(w))
    power = np.abs(np.fft.fft2((x-x.mean())*window))**2
    power[r==0] = 0
    bins = np.histogram2d(r.ravel(),a.ravel(),bins=(np.array([0,.025,.05,.1,.2,.35,.71]),np.linspace(0,np.pi,9)),weights=power.ravel())[0]
    return (bins/max(bins.sum(),1e-12)).ravel()


def image_features(gray):
    gray = np.clip(np.asarray(gray,float),0,1)
    assert gray.shape == (128,128)
    # Conservative global white-level normalization; retain unnormalized gray QC.
    white = max(float(np.quantile(gray,.95)),.5)
    g = np.clip(gray/white,0,1); dark=1-g; mask=g<.55
    sk = skeletonize(mask); degree=ndi.convolve(sk.astype(int),np.ones((3,3),int))-sk
    edt=ndi.distance_transform_edt(mask)
    valid=sk&(degree==2)&~ndi.binary_dilation(sk&(degree>2),iterations=2)
    yy,xx=np.nonzero(valid)
    # Local skeleton PCA: unlike intensity gradients, constant black interiors
    # have a defined orientation. Exclude short/isotropic neighborhoods.
    sy,sx=np.mgrid[-3:4,-3:4]; k=np.ones((7,7))
    mass=ndi.convolve(sk.astype(float),k)
    ex=ndi.correlate(sk.astype(float),sx)/np.maximum(mass,1)
    ey=ndi.correlate(sk.astype(float),sy)/np.maximum(mass,1)
    vx=ndi.correlate(sk.astype(float),sx*sx)/np.maximum(mass,1)-ex*ex
    vy=ndi.correlate(sk.astype(float),sy*sy)/np.maximum(mass,1)-ey*ey
    cov=ndi.correlate(sk.astype(float),sx*sy)/np.maximum(mass,1)-ex*ey
    anis=np.sqrt((vx-vy)**2+4*cov**2)/np.maximum(vx+vy,1e-9)
    keep=(mass[yy,xx]>=5)&(anis[yy,xx]>.35)
    yy,xx=yy[keep],xx[keep]
    angle=(.5*np.arctan2(2*cov[yy,xx],vx[yy,xx]-vy[yy,xx]))%np.pi
    width=2*edt[yy,xx]
    my,mx=np.nonzero(mask)
    box=max(np.ptp(my)+1,np.ptp(mx)+1) if len(my) else 128
    out={'ink':np.r_[hist(dark[mask],np.linspace(0,1,17)),mask.mean(),((gray>.02)&(gray<.98)).mean()],
         'fft_gray':spectrum(dark),'fft_binary':spectrum(mask.astype(float)),
         'layout':np.full(21,np.nan),'width':np.full(16,np.nan),
         'quill':np.full(192,np.nan),'hinge':np.full(288,np.nan),'profile':np.full(68,np.nan)}
    if len(my):
        grid=np.array([[mask[i:i+32,j:j+32].mean() for j in range(0,128,32)] for i in range(0,128,32)])
        out['layout']=np.r_[grid.ravel(),(np.ptp(mx)+1)/(np.ptp(my)+1),mx.mean()/128,my.mean()/128,mask.mean(),box/128]
    if len(width)>=8:
        edges=np.array([0,2,3,4,6,8,12,20,np.inf])
        out['width']=np.r_[hist(width,edges),hist(width/box,np.array([0,.02,.03,.04,.06,.08,.12,.2,np.inf]))]
        def joint(ws, bins):
            return np.histogram2d(angle,ws,bins=(np.linspace(0,np.pi,13),bins))[0].ravel()/len(ws)
        out['quill']=np.r_[joint(width,edges),joint(width/box,np.array([0,.02,.03,.04,.06,.08,.12,.2,np.inf]))]
        # Deterministic bounded sampling, symmetric u axis; image y points down.
        ids=np.linspace(0,len(width)-1,min(len(width),512)).astype(int)
        u=np.linspace(-1.5,1.5,33)
        y=yy[ids,None]+np.cos(angle[ids,None])*width[ids,None]/2*u
        x=xx[ids,None]-np.sin(angle[ids,None])*width[ids,None]/2*u
        profiles=ndi.map_coordinates(dark,[y,x],order=1,mode='constant',cval=0)
        # Reflection symmetrization avoids arbitrary sign of PCA eigenvectors.
        profiles=(profiles+profiles[:,::-1])/2
        local90=ndi.percentile_filter(dark,percentile=90,size=5)[yy,xx]
        out['profile']=np.r_[profiles.mean(0),profiles.std(0),local90.mean(),local90.std()]
    hs=[]
    contours=find_contours(mask.astype(float),.5)
    for step in (3,7):
        h=np.zeros((12,12))
        for points in contours:
            # Skip open/image-clipped and tiny contours; ordered closed contour.
            if len(points)<4*step or np.linalg.norm(points[0]-points[-1])>1.5: continue
            pts=points[:-1]
            v1=np.roll(pts,step,axis=0)-pts; v2=np.roll(pts,-step,axis=0)-pts
            a1=np.arctan2(v1[:,0],v1[:,1])%(2*np.pi)
            a2=np.arctan2(v2[:,0],v2[:,1])%(2*np.pi)
            h+=np.histogram2d(a1,a2,bins=(np.linspace(0,2*np.pi,13),)*2)[0]
        hs.append((h/max(h.sum(),1)).ravel() if h.sum() else np.full(144,np.nan))
    out['hinge']=np.concatenate(hs)
    out['qc']=np.array([mask.mean(),np.median(2*edt[sk]) if sk.any() else np.nan,len(width),((gray>.02)&(gray<.98)).mean()])
    return out


def latent_features(z):
    z=np.asarray(z,float);flat=z.reshape(4,-1); mu=flat.mean(1);sd=flat.std(1)
    centered=(flat-mu[:,None])/np.maximum(sd[:,None],1e-9)
    moments=np.stack([mu,sd,(centered**3).mean(1),(centered**4).mean(1)-3],1).ravel()
    grad=np.array([[np.mean(np.diff(ch,axis=a)**2) for a in (0,1)] for ch in z]).ravel()
    freq=[]
    for ch in z:
        p=spectrum(ch).reshape(6,8).sum(1)
        freq.extend([p[:2].sum(),p[2:4].sum(),p[4:].sum()])
    return np.r_[moments,grad,freq,np.cov(flat,bias=True).ravel()]
