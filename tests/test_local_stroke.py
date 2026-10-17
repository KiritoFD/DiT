import numpy as np
from tools.analysis.local_stroke_features import extract_local,triplet_margin


def test_distance_direction_tie_and_common_group_weights():
    f=np.array([[[0.],[0.]],[[.1],[100.]],[[2.],[1.]]])
    valid=np.array([[1,1],[1,0],[1,1]],bool)
    m,n=triplet_margin(f,np.array([[0,1,2],[0,2,1],[0,1,1]]),valid)
    assert n.tolist()==[1,1,1] and m[0]>0 and m[1]<0 and m[2]==0
    assert np.isclose(m[0],3.99)


def test_blank_has_no_groups_not_zero_distance_evidence():
    f=extract_local(np.ones((128,128)))
    assert f['counts'].sum()==0 and np.isnan(f['geometry']).all()


def test_rotating_stroke_rotates_group_but_not_aligned_profile():
    g=np.ones((128,128));g[60:68,16:112]=0
    h=extract_local(g);v=extract_local(g.T)
    hi=np.flatnonzero(np.isfinite(h['ink_edge']).all(1));vi=np.flatnonzero(np.isfinite(v['ink_edge']).all(1))
    assert len(hi) and len(vi) and set(hi)!=set(vi)
    assert np.allclose(h['ink_edge'][hi].mean(0),v['ink_edge'][vi].mean(0),atol=.03)


def test_no_image_mutation():
    g=np.ones((128,128));g[60:68,16:112]=0;before=g.copy()
    extract_local(g)
    assert np.array_equal(g,before)


def test_gpu_normalization_uses_training_rows_only():
    import pytest
    import torch
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from tools.analysis.run_local_stroke_audit import normalize_cuda
    x=np.array([[[0.,1.]],[[1.,2.]],[[2.,3.]]],np.float32)
    a=normalize_cuda(x,np.array([0,1]))
    x[2]=10000
    b=normalize_cuda(x,np.array([0,1]))
    assert np.allclose(a[:2],b[:2])
