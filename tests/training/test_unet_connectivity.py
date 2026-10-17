"""Cheap nonzero-output parity and transfer tests; no GPU training sweep."""
from dataclasses import replace

import pytest
import torch

from src.unet.io import Config,build
from src.unet.variants.transfer import load_expansion


@pytest.mark.parametrize('variant',['skip','decoder','std','residual'])
def test_extension_is_identity_and_has_gradient(variant):
    torch.set_num_threads(2)
    cfg=Config(channels=(32,64,64),max_steps=50000)
    base=build(cfg).eval()
    with torch.no_grad():base.output[-1].weight.normal_(std=.02)
    if variant=='residual':
        from src.unet.blocks import ResBlock
        with torch.no_grad():
            for block in base.modules():
                if isinstance(block,ResBlock):block.conv2.weight.normal_(std=.01)
    target=replace(cfg,**{'skip':dict(skip_gate_scales=(0,)),
                          'decoder':dict(decoder_res_blocks=(3,3,2)),
                          'std':dict(std_gate_scales=(0,)),
                          'residual':dict(residual_cond_gate=True)}[variant])
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
               'std':lambda:upgraded.std_gates[0],
               'residual':lambda:upgraded.encoder[0].blocks[0].residual_gate.weight}[variant]()
    assert parameter.grad is not None and parameter.grad.abs().sum()>0
    if variant=='skip':
        assert not upgraded.skip_gates[1].requires_grad and not upgraded.skip_gates[2].requires_grad
    broken=dict(ck['delta']);broken.pop('input.weight')
    with pytest.raises(ValueError,match='Unapproved'):
        load_expansion(upgraded,{'delta':broken},cfg,target)
    upgraded.load_state_dict(upgraded.state_dict(),strict=True)
    if variant=='residual':
        with pytest.raises(ValueError,match='residual gates'):
            load_expansion(build(cfg),{'delta':upgraded.state_dict()},target,cfg)


def test_baseline_and_inference_interventions():
    cfg=Config(channels=(32,64,64),max_steps=50000)
    m=build(cfg).eval()
    assert not any('gates' in k for k in m.state_dict())
    m.set_inference_scales(skip=(.8,1,1),std=(1,1,1))
    assert m.inference_skip_scales==(.8,1,1)
    m.train()
    with pytest.raises(RuntimeError):m.set_inference_scales()


@pytest.mark.parametrize('combined,gate',[(False,False),(True,False),(True,True)])
def test_writer_only_film_identity_gradient_and_strict_roundtrip(combined,gate,tmp_path):
    from src.unet.blocks import ResBlock
    from src.unet.io import load
    from dataclasses import asdict
    torch.set_num_threads(2)
    cfg=Config(channels=(32,64,64),max_steps=60000)
    base=build(cfg).eval()
    # A trained checkpoint has nonzero residual/output convolutions; a zero-head
    # test would conceal disconnected modulation gradients.
    with torch.no_grad():
        base.output[-1].weight.normal_(std=.02)
        for block in base.modules():
            if isinstance(block,ResBlock):block.conv2.weight.normal_(std=.01)
    target=replace(cfg,writer_film_rank=16,residual_cond_gate=gate,
                   **(dict(skip_gate_scales=(0,),std_gate_scales=(0,1,2),
                           decoder_res_blocks=(3,3,2)) if combined else {}))
    upgraded=build(target).eval()
    audit=load_expansion(upgraded,{'delta':base.state_dict()},cfg,target)
    assert any('.writer_film.' in key for key in audit['new_keys'])
    x=torch.randn(2,4,32,32);t=torch.tensor([.2,.8])
    kw=dict(y_callig=torch.tensor([0,1]),y_script=torch.tensor([0,1]),g=torch.rand(2,1,64,64))
    actual=upgraded(x,t,**kw)
    torch.testing.assert_close(actual,base(x,t,**kw),rtol=0,atol=0)
    actual.square().mean().backward()
    branch=upgraded.encoder[0].blocks[0].writer_film
    assert branch[-1].weight.grad.abs().sum()>0
    # A zero output projection learns first, then its input projection learns.
    with torch.no_grad():branch[-1].weight.add_(-.1*branch[-1].weight.grad)
    upgraded.zero_grad(set_to_none=True)
    upgraded(x,t,**kw).square().mean().backward()
    assert branch[0].weight.grad.abs().sum()>0
    checkpoint=tmp_path/'writer.pt'
    torch.save(dict(architecture=target.architecture,config=asdict(target),
                    delta=upgraded.state_dict(),ema=upgraded.state_dict()),checkpoint)
    restored,_,_=load(checkpoint,use_ema=False)
    torch.testing.assert_close(restored(x,t,**kw),upgraded(x,t,**kw),rtol=0,atol=0)
    without_writer=replace(target,writer_film_rank=0)
    with pytest.raises(ValueError,match='Writer FiLM'):
        load_expansion(build(without_writer),{'delta':upgraded.state_dict()},target,without_writer)


def test_writer_film_uses_dropped_writer_and_cfg_null():
    cfg=Config(channels=(32,64,64),writer_film_rank=16,max_steps=60000)
    model=build(cfg)
    seen=[]
    hook=model.encoder[0].blocks[0].writer_film.register_forward_pre_hook(
        lambda module,inputs:seen.append(inputs[0].detach().clone()))
    x=torch.randn(2,4,32,32);t=torch.tensor([.2,.8]);writers=torch.tensor([0,1])
    kw=dict(g=torch.rand(2,1,64,64),y_script=torch.tensor([0,1]))
    model.drop_writer=1.
    model(x,t,writers,**kw)
    null=model.writer.weight[model.num_writers].expand(2,-1)
    torch.testing.assert_close(seen[-1],null)
    model.eval();model.forward_with_cfg(x,t,writers,cfg_scale=2.,**kw)
    torch.testing.assert_close(seen[-2],model.writer(writers))
    torch.testing.assert_close(seen[-1],null)
    hook.remove()


def test_unit_residual_gate_does_not_block_zero_conv_startup():
    from src.unet.blocks import ResBlock
    from src.unet.variants.modulation import bounded_gain
    block=ResBlock(32,32)
    block.enable_residual_gate()
    x=torch.randn(2,32,8,8);condition=torch.randn(2,512)
    result=block(x,condition)
    torch.testing.assert_close(result,x,rtol=0,atol=0)
    result.square().mean().backward()
    assert block.conv2.weight.grad.abs().sum()>0
    with torch.no_grad():block.conv2.weight.add_(-.1*block.conv2.weight.grad)
    block.zero_grad(set_to_none=True)
    block(x,condition).square().mean().backward()
    assert block.residual_gate.weight.grad.abs().sum()>0
    torch.testing.assert_close(bounded_gain(torch.zeros(3)),torch.ones(3))
    gains=bounded_gain(torch.tensor([-1.,0.,1.]))
    assert ((gains>.5)&(gains<1.5)).all()


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
