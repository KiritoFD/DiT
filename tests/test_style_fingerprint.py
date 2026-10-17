import numpy as np
from tools.analysis.style_fingerprint_features import image_features, latent_features, matched_pairs


def test_blank_is_invalid_not_perfect_geometry():
    f=image_features(np.ones((128,128)))
    assert np.isnan(f['quill']).all()
    assert np.isfinite(f['fft_gray']).all()


def test_quill_distinguishes_horizontal_and_vertical_and_fixed_bins():
    g=np.ones((128,128));g[60:68,15:113]=0
    h=image_features(g);v=image_features(g.T)
    assert np.isclose(h['quill'][:96].sum(),1)
    assert np.isclose(v['quill'][:96].sum(),1)
    assert not np.allclose(h['quill'],v['quill'])
    assert np.allclose(h['width'],v['width'])
    assert len(h['profile'])==68 and np.isfinite(h['hinge']).all()


def test_latent_constant_is_finite_and_extractor_nonmutating():
    z=np.ones((4,32,32));original=z.copy()
    f=latent_features(z)
    assert f.shape==(52,) and np.isfinite(f).all()
    assert np.array_equal(z,original)


def test_matched_triplets_respect_content_writer_script_and_source():
    w=np.array(['A','A','B','A','B']);s=np.array(['K','K','K','K','X'])
    c=np.array(['one','two','one','three','one']);source=np.array(['q','q','n','other','n'])
    stats=np.array([[.2,4],[.2,4],[.2,4],[.205,4.1],[.2,4]])
    pairs,cross=matched_pairs(np.array([0,2,4]),np.array([1,3]),w,s,c,source,stats)
    assert pairs.tolist()==[[0,3,2]] and cross==1
    assert c[pairs[0,0]]!=c[pairs[0,1]]


def test_profile_reflection_invariant_and_black_center():
    g=np.ones((128,128));g[60:68,15:113]=0
    f=image_features(g)
    assert f['profile'][16]>.9
    assert np.allclose(f['profile'][:33],f['profile'][:33][::-1])


def test_cuda_latent_matches_cpu_reference():
    import pytest
    import torch
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from tools.analysis.style_fingerprint_gpu import extract
    rng=np.random.default_rng(89)
    z=rng.normal(size=(3,4,32,32)).astype(np.float16)
    actual=extract(z,batch=2)
    expected=np.stack([latent_features(x) for x in z])
    assert np.allclose(actual,expected,atol=2e-5,rtol=2e-5)
