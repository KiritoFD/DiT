"""Contour-Hinge (Bulacu & Schomaker 2007) and Quill (Brink et al. 2012)
calligraphy style features, CPU numpy/scipy. Also low-level nuisance features
for the E-G shortcut audit.

Deviations from the originals, frozen for eval_style_v2:
- Hinge: tangent-direction pairs at arc offsets +/-s along traced contours.
  The original uses outward edge-normal fragments; folding angles to [0,pi)
  makes the two versions a fixed cyclic relabeling of the joint histogram,
  which leaves any Hellinger-based verification unchanged.
- Quill: static stroke geometry only (orientation x width of ink), no stroke
  order or pressure claims. Width = 2x distance transform; orientation =
  structure-tensor minor direction of the smoothed ink map.
- Two hinge scales per image: s_short ~ 0.8x median stroke width (pixels),
  s_long ~ 3x s_short, both capped by contour length.
"""
import numpy as np
from scipy import ndimage as ndi

ANG_BINS=12
WIDTH_EDGES=np.geomspace(1.0,64.0,13)  # 12 log bins, 1..64 px
HINGE_DIM=ANG_BINS*ANG_BINS
FEATURE_VERSION='hq-1.0'
_NEIGH=[(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)]


def otsu_threshold(gray):
    """Upper edge of the last below-threshold bin, so mask=gray<t keeps that bin."""
    hist,_=np.histogram(gray,bins=256,range=(0.,1.))
    hist=hist.astype(float);total=hist.sum()
    if not total:return .5
    levels=np.arange(256)/256.
    sum_all=float((levels*hist).sum())
    w0=np.cumsum(hist);sum0=np.cumsum(levels*hist)
    valid=w0>0
    w1=total-w0;ok=valid&(w1>0)
    if not ok.any():return .5
    mu0=np.where(ok,sum0/np.maximum(w0,1e-12),0)
    mu1=np.where(ok,(sum_all-sum0)/np.maximum(w1,1e-12),0)
    between=np.where(ok,w0*w1*(mu0-mu1)**2,-1)
    return (float(np.argmax(between))+1)/256


def ink_parts(gray,inverted=False):
    t=otsu_threshold(gray)
    mask=(gray>t) if inverted else (gray<t)
    if mask.mean()>.5:mask=~mask
    if mask.mean()>.6 or mask.mean()<1e-4:  # degenerate image
        mask=np.zeros_like(mask)
    gray=(gray-gray.min())/max(gray.max()-gray.min(),1e-6)
    return mask,gray


def stroke_geometry(mask):
    dt=ndi.distance_transform_edt(mask)
    width=2.*dt
    smooth=ndi.gaussian_filter(mask.astype(np.float32),1.5)
    gx=ndi.sobel(smooth,axis=1);gy=ndi.sobel(smooth,axis=0)
    jxx=ndi.gaussian_filter(gx*gx,2.);jyy=ndi.gaussian_filter(gy*gy,2.)
    jxy=ndi.gaussian_filter(gx*gy,2.)
    theta=0.5*np.arctan2(2.*jxy,jxx-jyy)+np.pi/2  # stroke direction (ridges)
    theta=np.mod(theta,np.pi)
    return width,theta


def trace_contours(mask,min_len=8):
    """Moore-neighbor boundary tracing per 8-connected component."""
    lab,n=ndi.label(mask,structure=np.ones((3,3),int))
    H,W=mask.shape;contours=[]
    for ci in range(1,n+1):
        ys,xs=np.nonzero(lab==ci)
        if len(ys)<min_len:continue
        start=int(np.argmin(ys*W+xs));sy,sx=int(ys[start]),int(xs[start])
        contour=[(sy,sx)];cy,cx=sy,sx;back=6
        limit=8*len(ys)+8
        for _ in range(limit):
            for k in range(8):
                d=(back+1+k)%8
                ny,nx=cy+_NEIGH[d][0],cx+_NEIGH[d][1]
                if 0<=ny<H and 0<=nx<W and lab[ny,nx]==ci:
                    contour.append((ny,nx));back=(d+4)%8;cy,cx=ny,nx;break
            if (cy,cx)==(sy,sx):break
        if len(contour)>=min_len:contours.append(np.array(contour,dtype=np.int64))
    return contours


def _tangents(contour):
    d=np.diff(np.vstack([contour,contour[:1]]),axis=0).astype(float)
    seg=np.arctan2(d[:,0],d[:,1])
    z=np.exp(1j*seg)
    window=sum(np.roll(z,k) for k in (-2,-1,0,1,2))
    tan=np.angle(window)
    return np.mod(tan,np.pi)


def hinge_histogram(contour,s):
    n=len(contour)
    if n<2*s+6:return np.zeros((ANG_BINS,ANG_BINS))
    tan=_tangents(contour)
    idx=np.arange(n);jf=(idx+s)%n;jb=(idx-s)%n
    edges=np.linspace(0,np.pi,ANG_BINS+1)
    b1=np.clip(np.digitize(tan[jf],edges)-1,0,ANG_BINS-1)
    b2=np.clip(np.digitize(tan[jb],edges)-1,0,ANG_BINS-1)
    hist=np.zeros((ANG_BINS,ANG_BINS))
    np.add.at(hist,(b1,b2),1);np.add.at(hist,(b2,b1),1)
    return hist


def quill_histogram(theta,width,weight):
    edges_t=np.linspace(0,np.pi,ANG_BINS+1)
    tb=np.clip(np.digitize(theta,edges_t)-1,0,ANG_BINS-1)
    wb=np.clip(np.digitize(width,WIDTH_EDGES)-1,0,ANG_BINS-1)
    hist=np.zeros((ANG_BINS,ANG_BINS))
    np.add.at(hist,(tb,wb),weight)
    return hist


def nuisance_vector(gray,mask,width,theta):
    ys,xs=np.nonzero(mask)
    if len(ys)==0:return np.zeros(20)
    w=width[ys,xs];ink_pixels=w>0;w=w[ink_pixels]
    bbox=(xs.max()-xs.min()+1,ys.max()-ys.min()+1)
    cy,cx=ys.mean()/mask.shape[0],xs.mean()/mask.shape[1]
    edges_t=np.linspace(0,np.pi,13)
    oh,_=np.histogram(theta[ys,xs],edges_t)
    oh=oh/max(oh.sum(),1)
    er=ndi.binary_erosion(mask,np.ones((3,3),bool))
    boundary=mask&~er
    gy,gx=np.gradient(gray)
    sharp=float(np.hypot(gx,gy)[boundary].mean()) if boundary.any() else 0.
    return np.array([mask.mean(),np.percentile(w,25)/8,np.percentile(w,50)/8,
        np.percentile(w,75)/8,bbox[0]/mask.shape[1],bbox[1]/mask.shape[0],
        bbox[0]/max(bbox[1],1),cy,cx,sharp,*oh],dtype=np.float32)[:20]


def extract(gray,inverted=False):
    """Full feature dict for one gray image (float [0,1] array, ink=dark)."""
    mask,_=ink_parts(gray,inverted=inverted)
    width,theta=stroke_geometry(mask)
    ink=mask&(width>0)
    meta=dict(median_width=0.,s_short=0,s_long=0,ink_px=int(mask.sum()))
    if mask.sum()<50:
        return dict(hinge_s=np.zeros(HINGE_DIM),hinge_l=np.zeros(HINGE_DIM),
            quill=np.zeros(HINGE_DIM),nuisance=np.zeros(20),meta=meta)
    w_med=float(np.median(width[ink]))
    s_short=int(np.clip(round(.8*w_med),2,24));s_long=int(np.clip(3*s_short,3,72))
    contours=trace_contours(mask)
    hs=np.zeros((ANG_BINS,ANG_BINS));hl=np.zeros((ANG_BINS,ANG_BINS))
    for c in contours:
        hs+=hinge_histogram(c,s_short);hl+=hinge_histogram(c,s_long)
    ys,xs=np.nonzero(ink)
    qw=width[ys,xs];qt=theta[ys,xs]
    qweight=np.ones(len(ys),dtype=float)/max(len(ys),1)
    q=quill_histogram(qt,qw,qweight)
    meta.update(median_width=w_med,s_short=s_short,s_long=s_long)
    out=lambda h:(h.flatten()/max(h.sum(),1e-9)).astype(np.float32)
    return dict(hinge_s=out(hs),hinge_l=out(hl),quill=out(q),
        nuisance=nuisance_vector(gray,mask,width,theta),meta=meta)


def load_gray(path,size=256):
    from PIL import Image
    import numpy as np
    im=Image.open(path).convert('L')
    if im.size!=(size,size):im=im.resize((size,size),Image.Resampling.BICUBIC)
    return np.asarray(im,dtype=np.float32)/255.
