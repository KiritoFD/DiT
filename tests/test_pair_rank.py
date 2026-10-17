from types import SimpleNamespace
import numpy as np
import torch
import pytest
from src.train.pair_rank import PairMixtureSampler, rank_velocity_loss, ConditionalRank


def dataset():
    return SimpleNamespace(samples=[dict(script=s, character=c, calligrapher=w, aug='')
        for s in ('kai','xing') for c in ('X','Y','Z') for w in ('A','B','C')])


def test_pairs_and_shared_A_B_stream():
    data = dataset()
    a = PairMixtureSampler(data, 16, 50, seed=90)
    b = PairMixtureSampler(data, 16, 50, seed=90)
    ia, ib = list(a), list(b)
    assert ia == ib and len(ia) == 800
    for start in range(0, len(ia), 16):
        for j in (0,2):
            u,v = [data.samples[i] for i in ia[start+j:start+j+2]]
            assert (u['character'],u['script']) == (v['character'],v['script'])
            assert u['calligrapher'] != v['calligrapher']
    b.set_epoch(1)
    assert list(b) != ia


def test_rank_sign_and_both_gradients_are_live():
    target = torch.zeros(2,4,2,2)
    good = torch.full_like(target, .1, requires_grad=True)
    bad = torch.full_like(target, .5, requires_grad=True)
    loss, a, b = rank_velocity_loss(good,bad,target)
    reverse, _, _ = rank_velocity_loss(bad,good,target)
    assert loss < reverse
    loss.backward()
    assert torch.all(good.grad > 0)
    assert torch.all(bad.grad < 0)


def test_invalid_pair_coverage_fails_closed():
    with pytest.raises(ValueError,match='No legal'):
        PairMixtureSampler(SimpleNamespace(samples=[dict(script='kai',character='X',calligrapher='A')]),16,10)


def test_real_rank_hook_fixes_xt_and_std_preserves_rng(tmp_path):
    import csv
    import logging
    path=tmp_path/'train.csv'
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['img_id','script','character','calligrapher'])
        w.writeheader();w.writerows([dict(img_id=i,script='kai',character='X',calligrapher=str(i)) for i in range(4)])
    args=SimpleNamespace(diffusion_type='flow',pair_fraction=.5,w_critic_writer=0,w_critic_away=0,
        strong_style_mode='off',data_csv=str(path),global_seed=90,rank_t_min=.4,rank_t_max=.8,
        rank_margin=.01,rank_tau=.1,strong_style_max_weight=10,rank_gradient_ratio=.075)
    calls=[]
    class Toy(torch.nn.Module):
        def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.3))
        def forward(self,x,t,**kw):
            calls.append((x.detach().clone(),t.detach().clone(),kw))
            return self.weight*x+kw['y_callig'][:,None,None,None]*self.weight
    model=Toy();hook=ConditionalRank(args,'cpu',logging.getLogger('test'))
    batch=dict(img_id=torch.arange(4),std_soft=torch.full((4,1,64,64),128,dtype=torch.uint8),
        y_callig_raw=torch.arange(4),y_pair=torch.arange(4),y_script=torch.zeros(4,dtype=torch.long))
    state=torch.get_rng_state().clone()
    value=hook(model,torch.ones(4,4,32,32),None,{'y_char':torch.zeros(4,dtype=torch.long)},batch,None,1,fm_norm=torch.tensor(1.))
    assert torch.equal(state,torch.get_rng_state())
    assert model.training
    assert torch.equal(calls[0][0],calls[1][0])
    assert torch.equal(calls[0][1],calls[1][1])
    assert torch.equal(calls[0][2]['g'],calls[1][2]['g'])
    assert calls[0][2]['g'].max()==pytest.approx(128/255)
    assert calls[1][2]['y_callig'].tolist()==[1,0]
    value.backward()
    assert model.weight.grad is not None and model.weight.grad.abs()>0
