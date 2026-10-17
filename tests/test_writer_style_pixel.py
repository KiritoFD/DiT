import inspect
import numpy as np
import pytest
import torch
from src.model.writer_style_pixel import WriterStylePixel,augment_gray
from src.utils.writer_quartets import WriterQuartets,cross_loss


def test_model_input_boundary_size_and_gradients():
    torch.set_num_threads(2);m=WriterStylePixel(3)
    assert list(inspect.signature(m.forward).parameters)==['gray']
    count=sum(p.numel() for p in m.parameters());assert 2_000_000<count<5_000_000
    o=m(torch.rand(4,1,128,128));assert o['embedding'].shape==(4,256)
    assert torch.allclose(o['embedding'].norm(dim=1),torch.ones(4),atol=1e-5)
    loss,metrics=cross_loss(o['embedding'],np.array(['one','two','one','two']))
    loss.backward();assert m.stem[0].weight.grad.abs().sum()>0
    assert metrics['same_char_hard_fraction']==1


def test_full_quartets_train_only_symmetric_and_deterministic():
    labels=np.array([0,0,1,1,0,1]);scripts=np.zeros(6,int)
    chars=np.array(['x','y','x','y','z','z']);sources=np.array(['a','b','c','d','a','c'])
    samplers=[WriterQuartets(np.arange(4),labels,scripts,chars,sources,seed=4) for _ in range(2)]
    a,metrics=samplers[0].sample(40);b,_=samplers[1].sample(40)
    assert np.array_equal(a,b) and set(a)<=set(range(4)) and metrics['full_quartet_fraction']==1
    for ids in a.reshape(-1,4):
        assert labels[ids[0]]==labels[ids[1]]!=labels[ids[2]]==labels[ids[3]]
        assert chars[ids[0]]==chars[ids[2]]!=chars[ids[1]]==chars[ids[3]]


def test_fallback_hard_coverage_is_not_mislabeled():
    labels=np.array([0,0,1,1]);chars=np.array(['x','y','x','z'])
    sampler=WriterQuartets(np.arange(4),labels,np.zeros(4,int),chars,np.array(['a']*4))
    ids,metrics=sampler.sample(4);assert metrics['full_quartet_fraction']==0
    e=torch.nn.functional.normalize(torch.randn(4,8),dim=-1).requires_grad_(True)
    loss,m=cross_loss(e,chars[ids]);assert m['same_char_hard_fraction']==.5
    loss.backward();assert torch.isfinite(e.grad).all()


def test_cross_loss_prefers_writer_not_character():
    correct=torch.tensor([[1.,0],[1.,0],[0,1.],[0,1.]])
    character=torch.tensor([[1.,0],[0,1.],[1.,0],[0,1.]])
    chars=np.array(['x','y','x','y'])
    assert cross_loss(correct,chars)[0]<cross_loss(character,chars)[0]


def test_augmentation_preserves_original_and_white_padding():
    x=torch.ones(4,1,128,128);original=x.clone();y=augment_gray(x)
    assert torch.equal(x,original) and y.shape==x.shape
    assert torch.isfinite(y).all() and y.min()>=0 and y.max()<=1 and y.mean()>.9
