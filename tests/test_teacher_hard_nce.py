import torch
from src.loss.style_encoder_loss import same_character_hard_nce


def test_explicit_hard_nce_rewards_cross_character_positive_over_same_character_negative():
    a=torch.tensor([[1.,0.]],requires_grad=True)
    p=torch.tensor([[1.,0.]])
    n=torch.tensor([[0.,1.]])
    valid=torch.tensor([True])
    good=same_character_hard_nce(a,p,n,valid)
    bad=same_character_hard_nce(a,n,p,valid)
    assert good.item()<.001 and bad.item()>9
    bad.backward()
    assert torch.isfinite(a.grad).all() and a.grad.abs().sum()>0


def test_held_out_pixel_statistics_do_not_change_training_pair_selection():
    import numpy as np
    from src.utils.style_encoder_data import CrossCharacterSampler
    slots=np.repeat(np.arange(3),4);scripts=np.zeros(12,dtype=int)
    chars=np.array(['x','y','z','q']*3);sources=np.array(['a','b','c','test']*3)
    train=np.array([i for i in range(12) if i%4!=3])
    stats=np.ones((12,2));stats[:,0]=.1
    altered=stats.copy();altered[[3,7,11]]=999
    a=CrossCharacterSampler(train,slots,scripts,chars,sources,statistics=stats,balance_sources=True)
    b=CrossCharacterSampler(train,slots,scripts,chars,sources,statistics=altered,balance_sources=True)
    assert np.array_equal(a.sample(64)[0],b.sample(64)[0])
