import numpy as np
import pytest
import torch
from tools.analysis.swd_style_stats import projection_directions,train_normalization,mixture_quantiles,swd_from_quantiles,summarize_distances
from tools.analysis.swd_style_stats import overlapping_membership_bootstrap,global_permutation_diagnostic


def test_projection_seed_and_unit_norm():
    a=projection_directions(36)
    assert np.array_equal(a,projection_directions(36))
    assert np.allclose(np.linalg.norm(a,axis=0),1)


def test_normalization_uses_only_train_rows():
    x=np.arange(24,dtype=np.float32).reshape(4,3,2)
    a=train_normalization(x,[0,1]);x[2:]=1e8;b=train_normalization(x,[0,1])
    for aa,bb in zip(a,b):assert np.array_equal(aa,bb)


def test_mixture_is_not_average_quantile():
    x=np.array([0,2,10,12],np.float32).reshape(2,1,1,2)
    q=mixture_quantiles(x,[[0,1]],'cpu')
    assert q.flatten().tolist()==[0,2,10,12]
    assert not torch.equal(q.flatten(),torch.tensor([5.,5.,7.,7.]))


def test_swd_symmetry_identity_and_known_translation():
    a=torch.tensor([[[[0.,1.,2.]]]])
    assert swd_from_quantiles(a,a).item()==0
    assert swd_from_quantiles(a,a+3).item()==pytest.approx(9)
    assert torch.equal(swd_from_quantiles(a,a+3),swd_from_quantiles(a+3,a))


def test_mixture_invariant_to_character_and_patch_order():
    rng=np.random.default_rng(4);x=rng.normal(size=(3,1,4,8)).astype(np.float32)
    assert torch.equal(mixture_quantiles(x,[[0,1,2]],'cpu'),mixture_quantiles(x[:,:,:,::-1].copy(),[[2,0,1]],'cpu'))


def test_score_direction_is_not_flipped():
    trials=[{'w':0,'v':1,'source_block':'s','test_chars':['c']}]*4
    result=summarize_distances(np.ones((4,2))*2,np.ones((4,2)),trials)
    assert result['R']==.5
    assert result['two_afc']==0


def test_overlap_bootstrap_shares_character_memberships():
    r=overlapping_membership_bootstrap(np.ones(3),np.ones(3)*2,np.ones(3),[['a','b'],['b','c'],['a','c']])
    assert r['n_clusters']==3
    assert r['R_ci95']==pytest.approx([2,2])
    assert r['two_afc_ci95']==pytest.approx([1,1])


def test_global_permutation_respects_script_and_reports_coverage():
    distances=np.array([[[0,2,np.nan],[2,0,np.nan]]],np.float32)
    result=global_permutation_diagnostic(distances,[{'w':0,'v':1}],[0,0,1],draws=20)
    assert result['draws_with_ge80pct_coverage']==20
    assert result['coverage_quantiles']==[1,1,1]
    assert result['mean_observed_minus_permuted_on_same_valid_trials']>0
