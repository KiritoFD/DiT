import numpy as np
from tools.analysis.allograph_stats import make_trials,measure,distance


def fixture():
    # Two writers, eight chars; independent source, same character distributions.
    labels=np.repeat([0,1],8);chars=np.tile(np.array(list('abcdefgh')),2)
    return {'labels':labels,'si':np.zeros(16,int),'char':chars,'source':np.tile(np.array(['A']*4+['B']*4),2),
        'slots':[('w','K'),('v','K')], 'splits':{'train':np.r_[0:4,8:12],'test_char':np.r_[4:8,12:16],'test_source':np.r_[4:8,12:16]}}


def test_character_matched_groups_and_no_replacement():
    d=fixture();trials,_=make_trials(d,4,source_holdout=True,repeats=2)
    assert len(trials)==2
    for t in trials:
        a,b,c,e=map(np.array,t['ids'])
        assert set(d['char'][a])==set(d['char'][c]) and not set(d['char'][a])&set(d['char'][b])
        assert all(len(set(ids))==4 for ids in t['ids'])
    assert make_trials(d,8)[0]==[]


def test_writer_signal_above_content_and_distance_sign():
    d=fixture();trials,_=make_trials(d,2,repeats=5)
    f=np.array([[1.,.01]]*8+[[.01,1.]]*8)
    result=measure(f,trials,'shape');assert result['R']>1.2 and result['two_afc']==1
    assert distance(np.array([1,0]),np.array([1,0]),'hist')==0
    assert distance(np.array([1,0]),np.array([0,1]),'hist')>.99


def test_codebook_fit_cannot_see_heldout_tokens():
    from tools.analysis.run_allograph_audit import fit_histograms
    rng=np.random.default_rng(89);tokens=rng.normal(size=(512,4)).astype(np.float16)
    owners=np.repeat(np.arange(64),8);train=np.arange(32)
    _,a,info=fit_histograms(tokens,owners,train,64)
    changed=tokens.copy();changed[owners>=32]=1000
    _,b,_=fit_histograms(changed,owners,train,64)
    assert np.array_equal(a,b) and set(info['fit_owner_indices'])==set(train)
