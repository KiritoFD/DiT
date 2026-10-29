"""Synthetic smoke tests for Hinge/Quill feature extraction. Run anywhere."""
import numpy as np
from . import features_hq as hq


def bar_image(angle_deg,width=6,size=256):
    img=np.ones((size,size),dtype=np.float32)
    cy,cx=size/2,size/2
    a=np.deg2rad(angle_deg);dy,dx=np.sin(a),np.cos(a)
    for y in range(size):
        for x in range(size):
            ry=y-cy;rx=x-cx
            along=ry*dy+rx*dx;perp=-ry*dx+rx*dy
            if abs(along)<90 and abs(perp)<width/2:img[y,x]=0.0
    return img


def test_horizontal_bar():
    f=hq.extract(1.-bar_image(0))  # ink=dark -> invert bright bar
    q=f['quill'].reshape(hq.ANG_BINS,hq.ANG_BINS)
    theta_mass=q.sum(1)
    top_theta=int(np.argmax(theta_mass))
    edges=np.linspace(0,np.pi,hq.ANG_BINS+1)
    center=(edges[top_theta]+edges[top_theta+1])/2
    expect=0.
    assert abs(((center-expect+np.pi/2)%np.pi)-np.pi/2)<np.pi/hq.ANG_BINS, f'horizontal bar theta bin center {center:.2f}'
    assert theta_mass[top_theta]>0.25, theta_mass


def test_vertical_bar():
    f=hq.extract(1.-bar_image(90))
    q=f['quill'].reshape(hq.ANG_BINS,hq.ANG_BINS)
    theta_mass=q.sum(1)
    top=int(np.argmax(theta_mass))
    edges=np.linspace(0,np.pi,hq.ANG_BINS+1)
    center=(edges[top]+edges[top+1])/2
    assert abs(center-np.pi/2)<np.pi/hq.ANG_BINS, f'vertical center {center:.2f}'


def test_width_bins():
    thick=hq.extract(1.-bar_image(0,width=12))['quill']
    thin=hq.extract(1.-bar_image(0,width=3))['quill']
    thick_w=thick.reshape(hq.ANG_BINS,hq.ANG_BINS).sum(0)
    thin_w=thin.reshape(hq.ANG_BINS,hq.ANG_BINS).sum(0)
    assert np.argmax(thick_w)>np.argmax(thin_w), (thick_w,thin_w)


def test_determinism_and_hinge():
    img=1.-bar_image(30)
    a=hq.extract(img);b=hq.extract(img)
    for key in ('hinge_s','hinge_l','quill','nuisance'):
        assert np.array_equal(a[key],b[key]), key
    # straight bar: hinge pairs concentrate near (0,0) i.e. parallel tangents
    hs=a['hinge_s'].reshape(hq.ANG_BINS,hq.ANG_BINS)
    parallel=hs.trace()+np.trace(np.fliplr(hs))
    assert parallel>0.35, parallel


def test_circle_curvature_spread():
    size=256;yy,xx=np.mgrid[:size,:size]
    disk=((yy-size/2)**2+(xx-size/2)**2)<80**2
    img=np.where(disk,0.,1.).astype(np.float32)
    f=hq.extract(img)
    hs=f['hinge_s'].reshape(hq.ANG_BINS,hq.ANG_BINS)
    straight=hs.trace()+np.trace(np.fliplr(hs))
    assert straight<0.55, straight  # curved contour spreads beyond parallel bins


def test_real_image_shape():
    rng=np.random.default_rng(0)
    img=np.clip(0.9-0.8*np.maximum(np.zeros((256,256)),0),0,1).astype(np.float32)
    img[100:150,40:220]=0.05;img[60:200,120:140]=0.05  # cross stroke
    f=hq.extract(img)
    assert f['hinge_s'].shape==(hq.HINGE_DIM,) and abs(f['hinge_s'].sum()-1)<1e-5
    assert abs(f['quill'].sum()-1)<1e-5
    assert len(f['nuisance'])==20 and np.isfinite(f['nuisance']).all()
    assert f['meta']['s_short']>=2 and f['meta']['s_long']>=f['meta']['s_short']


if __name__=='__main__':
    for name,fn in sorted({k:v for k,v in globals().items() if k.startswith('test_')}.items()):
        fn();print('PASS',name)
    print('ALL HQ SMOKE TESTS PASS')
