import numpy as np
import torch
from src.model.style_encoder import build_style_encoder,augment_style_inputs
from src.utils.style_encoder_data import CrossCharacterSampler


def test_cosine_classifier_uses_normalized_signed_embedding():
    model=build_style_encoder(26,3,'local_stats').eval()
    z=torch.randn(3,4,32,32)
    g=torch.rand(3,1,64,64)
    out=model(z,g,torch.tensor([0,1,2]))
    assert out['embedding'].shape==(3,256)
    assert torch.allclose(out['embedding'].norm(dim=1),torch.ones(3),atol=1e-5)
    expected=16*torch.nn.functional.linear(out['embedding'],torch.nn.functional.normalize(model.class_weight,dim=-1))
    assert torch.allclose(out['slot_logits'],expected,atol=1e-5)
    assert (out['embedding']<0).any()


def test_script_cannot_supply_identity_to_embedding_and_blank_std_is_finite():
    model=build_style_encoder(26,3,'local_stats').eval()
    z=torch.randn(2,4,32,32)
    g=torch.zeros(2,1,64,64)
    a=model(z,g,torch.tensor([0,0]))
    b=model(z,g,torch.tensor([2,2]))
    assert torch.equal(a['embedding'],b['embedding'])
    assert torch.isfinite(a['embedding']).all()


def test_new_encoder_gradient_and_checkpoint_roundtrip():
    model=build_style_encoder(26,3,'local_stats').eval().requires_grad_(False)
    other=build_style_encoder(26,3,'local_stats').eval()
    other.load_state_dict(model.state_dict(),strict=True)
    z=torch.randn(2,4,32,32,requires_grad=True)
    g=torch.rand(2,1,64,64);s=torch.tensor([0,1])
    result=model(z,g,s)['embedding']
    assert torch.equal(result,other(z,g,s)['embedding'])
    result[:,0].sum().backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum()>0


def test_pair_augmentation_is_finite_and_does_not_modify_cached_inputs():
    z=torch.randn(8,4,32,32);g=torch.rand(8,1,64,64)
    zz,gg=z.clone(),g.clone()
    a,b=augment_style_inputs(z,g)
    assert torch.equal(z,zz) and torch.equal(g,gg)
    assert a.shape==z.shape and b.shape==g.shape
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    assert b.min()>=0 and b.max()<=1


def test_source_balanced_anchors_do_not_follow_image_count_imbalance():
    slots=[];scripts=[];chars=[];sources=[]
    for s in range(2):
        for source,n in [('large',400),('small',4)]:
            for j in range(n):
                slots.append(s);scripts.append(0);chars.append(str(j%2));sources.append(source)
    slots,scripts,chars,sources=map(np.asarray,(slots,scripts,chars,sources))
    sampler=CrossCharacterSampler(np.arange(len(slots)),slots,scripts,chars,sources,balance_sources=True)
    ids,_=sampler.sample(1000)
    fraction=np.mean(sources[ids[:1000]]=='small')
    assert .4<fraction<.6
