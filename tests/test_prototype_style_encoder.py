import numpy as np
import pytest
import torch
from src.model.style_encoder import StyleEncoder
from src.loss.style_encoder_loss import cross_character_supcon, std_separation
from src.utils.style_encoder_data import CrossCharacterSampler, make_splits
from tools.train_prototype_style_encoder import source_balanced_prototypes, retrieval_metrics


def synthetic():
    rows = [(s,c,f'source{w}') for s in range(4) for c in range(60) for w in range(3)]
    return (np.array([r[0] for r in rows]),np.array([r[0]//2 for r in rows]),
            np.array([str(r[1]) for r in rows]),np.array([r[2] for r in rows]))


def test_split_is_disjoint_and_reserves_official_characters():
    slots,_,chars,sources = synthetic()
    split = make_splits(chars,slots,sources,{'0','1'})
    sets = [set(v) for v in split.values()]
    assert all(not (a & b) for i,a in enumerate(sets) for b in sets[i+1:])
    assert not set(chars[split['train']]) & set(chars[split['val_char']])
    assert not set(chars[split['train']]) & set(chars[split['test_char']])
    assert not set(chars[split['train']]) & {'0','1'}
    train_groups = set(zip(slots[split['train']],sources[split['train']]))
    assert not train_groups & set(zip(slots[split['test_source']],sources[split['test_source']]))
    again = make_splits(chars,slots,sources,{'0','1'})
    assert all(np.array_equal(split[k],again[k]) for k in split)


def test_sampler_explicit_cross_char_same_script_pairs():
    slots,scripts,chars,sources = synthetic()
    sampler = CrossCharacterSampler(np.arange(len(slots)),slots,scripts,chars,sources)
    ids, stats = sampler.sample(32)
    a,p,n = np.split(ids,3)
    assert np.all(slots[a]==slots[p]) and np.all(chars[a]!=chars[p])
    assert np.all(slots[a]!=slots[n]) and np.all(scripts[a]==scripts[n])
    assert np.all(chars[a]==chars[n])
    assert stats['hard_negative_fraction']==1


def test_supcon_all_batch_gradients_and_invalid_pairs():
    x = torch.randn(8,16,requires_grad=True)
    embedding = torch.nn.functional.normalize(x,dim=-1)
    slots = torch.tensor([0,0,1,1,2,2,3,3])
    scripts = slots//2
    chars = torch.tensor([0,1]*4)
    loss, valid = cross_character_supcon(embedding,slots,scripts,chars)
    loss.backward()
    assert valid.item()==1 and torch.isfinite(x.grad).all()
    assert (x.grad.abs().sum(1)>0).all()
    with pytest.raises(ValueError):
        cross_character_supcon(embedding,slots,scripts,torch.zeros(8,dtype=torch.long))


def test_std_margin_direction():
    a = torch.tensor([[1.,0.]])
    b = torch.tensor([[0.,1.]])
    assert std_separation(a,a,b).item()==0
    assert std_separation(a,b,a).item()>1


def test_encoder_normalized_and_differentiable_to_generated_latent():
    model = StyleEncoder(26,3).eval().requires_grad_(False)
    z = torch.randn(2,4,32,32,requires_grad=True)
    output = model(z,torch.rand(2,1,64,64),torch.tensor([0,1]))
    assert output['embedding'].shape==(2,256)
    assert torch.allclose(output['embedding'].norm(dim=1),torch.ones(2),atol=1e-5)
    output['embedding'][:,0].sum().backward()
    assert z.grad.abs().sum()>0


def test_prototypes_equal_source_weight_not_image_count():
    features = np.array([[1.,0.]]*9+[[0.,1.]])
    result = source_balanced_prototypes(features,np.zeros(10,int),np.array(['a']*9+['b']),1)
    assert np.allclose(result[0],np.array([1.,1.])/np.sqrt(2))


def test_retrieval_excludes_same_character_shortcut():
    query = np.array([[1.,0.]])
    gallery = np.array([[1.,0.],[0.,1.],[.8,.2]])
    result = retrieval_metrics(query,gallery,np.array([0]),np.array([1,0,1]),
        np.array([0]),np.zeros(3,int),np.array(['same']),np.array(['same','other','different']),
        np.eye(2),np.zeros(2,int))
    assert result['n_valid']==1
    assert result['cross_char_map']==.5
