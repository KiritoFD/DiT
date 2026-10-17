"""Cheap nonzero-output parity and transfer tests; no GPU training sweep."""
from dataclasses import replace

import pytest
import torch

from src.unet.io import Config,build
from src.unet.variants.transfer import load_expansion


@pytest.mark.parametrize('variant',['skip','decoder','std'])
def test_extension_is_identity_and_has_gradient(variant):
    torch.set_num_threads(2)
    cfg=Config(channels=(32,64,64),max_steps=50000)
    base=build(cfg).eval()
    with torch.no_grad():base.output[-1].weight.normal_(std=.02)
    target=replace(cfg,**{'skip':dict(skip_gate_scales=(0,)),
                          'decoder':dict(decoder_res_blocks=(3,3,2)),
                          'std':dict(std_gate_scales=(0,))}[variant])
    upgraded=build(target).eval()
    ck={'delta':base.state_dict()}
    audit=load_expansion(upgraded,ck,cfg,target)
    assert audit['new_keys']
    x=torch.randn(2,4,32,32);t=torch.tensor([200.,800.]);g=torch.rand(2,1,64,64)
    kw=dict(y_callig=torch.tensor([0,1]),y_script=torch.tensor([0,1]),g=g)
    expected=base(x,t,**kw);actual=upgraded(x,t,**kw)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    actual.square().mean().backward()
    parameter={'skip':lambda:upgraded.skip_gates[0],
               'decoder':lambda:upgraded.decoder[0].blocks[2].conv2.weight,
               'std':lambda:upgraded.std_gates[0]}[variant]()
    assert parameter.grad is not None and parameter.grad.abs().sum()>0
    if variant=='skip':
        assert not upgraded.skip_gates[1].requires_grad and not upgraded.skip_gates[2].requires_grad
    broken=dict(ck['delta']);broken.pop('input.weight')
    with pytest.raises(ValueError,match='Unapproved'):
        load_expansion(upgraded,{'delta':broken},cfg,target)
    upgraded.load_state_dict(upgraded.state_dict(),strict=True)


def test_baseline_and_inference_interventions():
    cfg=Config(channels=(32,64,64),max_steps=50000)
    m=build(cfg).eval()
    assert not any('gates' in k for k in m.state_dict())
    m.set_inference_scales(skip=(.8,1,1),std=(1,1,1))
    assert m.inference_skip_scales==(.8,1,1)
    m.train()
    with pytest.raises(RuntimeError):m.set_inference_scales()


def test_deep_encoder_cannot_be_silently_warm_started():
    cfg=Config(channels=(32,64,64),max_steps=50000)
    deeper=replace(cfg,num_res_blocks=3)
    with pytest.raises(ValueError,match='from scratch'):
        load_expansion(build(deeper),{'delta':build(cfg).state_dict()},cfg,deeper)


def test_threshold_audit_distinguishes_dots_from_broken_strokes():
    import numpy as np
    from tools.analysis.unet_connectivity.components import topology
    gt=np.ones((24,24));gt[10:12,2:20]=0
    dotted=gt.copy();dotted[0,0]=0
    assert topology(dotted,gt,min_area=0)['frag']==2
    assert topology(dotted,gt,min_area=2)['frag']==1
    broken=gt.copy();broken[:,10:12]=1
    assert topology(broken,gt,min_area=4)['frag']==2


def test_raw_and_intervention_protocol_identity(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import src.eval.in_mem_eval as evaluator
    (tmp_path/'weights.bin').write_bytes(b'identity-test')
    csv=tmp_path/'data.csv';csv.write_text('image_path\nexample.png\n')
    args=SimpleNamespace(eval_vae_path=str(tmp_path),eval_use_ema=False,std_soft_eval_sigma=.65)
    monkeypatch.setattr(evaluator,'_DIFF',SimpleNamespace(sampler='heun',sde_gamma=0.))
    model=torch.nn.Linear(2,2)
    raw=evaluator._build_eval_request(args,model,20100,'test',1,1.,50,.18215,str(csv),'test')
    assert not raw.ema
    args.eval_protocol_extra={'case':'skip08','skip':[.8,1.,1.]}
    changed=evaluator._build_eval_request(args,model,20100,'test',1,1.,50,.18215,str(csv),'test')
    assert changed.eval_id()!=raw.eval_id()
    assert changed.extra['diagnostic']['case']=='skip08'


def test_std_probe_removes_its_hooks(monkeypatch):
    from types import SimpleNamespace
    import src.eval.in_mem_eval as evaluator
    from tools.analysis.unet_connectivity.sample import std_ratios
    model=build(Config(channels=(32,64,64),max_steps=50000)).eval()
    cache=dict(conds=[(i,i) for i in range(4)],hier_conds=([],[],[0,1,0,1]),
               noise=torch.randn(4,4,32,32),skels_latent=torch.rand(4,1,64,64))
    monkeypatch.setattr(evaluator,'_get_cache',lambda *args,**kw:cache)
    result=std_ratios(model,SimpleNamespace(eval_csv='unused',img_root='unused'),'cpu')
    assert [r['scale'] for r in result['scales']]==[32,16,8]
    assert all(r['ratio']>0 for r in result['scales'])
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())
