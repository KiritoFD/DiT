import numpy as np
import pytest
from tools.analysis.accept_v89_encoder import wrong_std_lookup, same_char_2afc


def test_wrong_std_preserves_script_and_changes_character():
    scripts=np.array(['a','a','b','b'])
    chars=np.array(['x','y','x','y'])
    result=wrong_std_lookup(scripts,chars,scripts,chars)
    assert np.all(scripts[result]==scripts)
    assert np.all(chars[result]!=chars)


def test_wrong_std_cannot_silently_use_same_character():
    with pytest.raises(ValueError):
        wrong_std_lookup(['a'],['x'],['a'],['x'])


def test_same_char_2afc_counts_ties_and_skips_unmatched_characters():
    features=np.array([[1.,0.],[1.,0.],[0.,1.]])
    result=same_char_2afc(features,np.array([0,1,0]),np.zeros(3),np.array(['x','x','y']),np.eye(2))
    assert result['n_valid']==2 and result['n_skipped']==1
    assert result['accuracy']==.5
