import inspect
import numpy as np
import pytest
import torch
from src.model.writer_style_latent import WriterStyleLatent,augment_latent
from src.utils.writer_quartets import cross_loss


def test_latent_only_multiscale_and_nonzero_style_gradient():
    torch.set_num_threads(2);model=WriterStyleLatent(23)
    assert list(inspect.signature(model.forward).parameters)==['z']
    assert 500_000<sum(p.numel() for p in model.parameters())<2_000_000
    z=torch.randn(4,4,32,32,requires_grad=True);out=model(z)
    assert out['embedding'].shape==(4,256) and out['slot_logits'].shape==(4,23)
    assert torch.allclose(out['embedding'].norm(dim=1),torch.ones(4),atol=1e-5)
    loss,_=cross_loss(out['embedding'],np.array(['a','b','a','b']))
    loss.backward();assert torch.isfinite(z.grad).all() and z.grad.norm()>0


def test_frozen_model_retains_input_gradient_and_rejects_std():
    model=WriterStyleLatent().eval().requires_grad_(False);z=torch.randn(4,4,32,32,requires_grad=True)
    out=model(z);loss,_=cross_loss(out['embedding'],np.array(['a','b','a','b']));loss.backward()
    assert z.grad.norm()>0 and all(p.grad is None for p in model.parameters())
    with pytest.raises(TypeError):model(z,g=torch.ones(4,1,64,64))
    with pytest.raises(ValueError):model(torch.randn(4,4,64,64))


def test_latent_augmentation_nonmutating_seeded_no_scale_division():
    z=torch.ones(4,4,32,32)*3;before=z.clone();torch.manual_seed(89);a=augment_latent(z)
    torch.manual_seed(89);b=augment_latent(z)
    assert torch.equal(z,before) and torch.equal(a,b)
    assert 2.7<float(a.mean())<3.3 and torch.isfinite(a).all()
