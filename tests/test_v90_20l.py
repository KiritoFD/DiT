import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from src.model.dit import DiT_2Cond,DiT_2Cond_models
from src.model.dit_conditioning import EmbedGlyphCfg,embed_glyph
from src.utils.std_shape_aug import StdShapeAugment,low_frequency_field,mixture,guards,width_jitter
from tools.analysis.prepare_v90_20l import configuration


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(4)
    yield
    torch.set_num_threads(old)


def inputs(n=4):
    raw=torch.zeros(n,1,256,256,dtype=torch.uint8)
    raw[:,:,40:216,116:140]=255;raw[:,:,105:129,40:216]=255
    clean=F.interpolate(raw.float()/255,size=(64,64),mode='area')
    return raw,clean


def tiny(**kw):
    return DiT_2Cond(depth=2,hidden_size=64,num_heads=4,patch_size=2,
        num_calligraphers=10,num_characters=12,use_char_cond=False,std_soft_cond=True,
        use_glyph_cond=True,glyph_in_channels=1,content_cond='std',condition_fusion='factorized_cat',**kw)


@pytest.mark.parametrize('step,expected',[(0,(.3,.35,.3,.05)),(10000,(.2,.25,.4,.15)),(150000,(.25,.3,.35,.1))])
def test_phase_mixture(step,expected):assert mixture(step)==expected


def test_elastic_is_zero_mean_low_frequency_and_bounded():
    field=low_frequency_field(4,256,256,torch.tensor([0.,4.,8.,10.]),'cpu',torch.Generator().manual_seed(90))
    assert field.mean((2,3)).abs().max()<1e-5
    assert torch.all(field.square().sum(1).sqrt().flatten(1).amax(1)<=torch.tensor([0.,4.,8.,10.])+1e-5)
    assert (field[:,:,:,1:]-field[:,:,:,:-1]).abs().max()<1


@pytest.mark.parametrize('tier',[0,1,2,3])
def test_shape_aug_determinism_clean_exact_and_range(tier):
    raw,clean=inputs();saved=raw.clone()
    aug=StdShapeAugment()
    a,stats=aug(raw,clean,force_tier=tier,generator=torch.Generator().manual_seed(9))
    b,_=aug(raw,clean,force_tier=tier,generator=torch.Generator().manual_seed(9))
    assert torch.equal(a,b) and torch.equal(raw,saved)
    assert a.shape==clean.shape and torch.isfinite(a).all() and a.min()>=0 and a.max()<=1
    if tier==0:assert torch.equal(a,clean)
    else:assert not torch.equal(a,clean)
    good,_=guards(a,clean,stats['clipped_fraction'])
    assert good.all()


def test_fake_high_resolution_is_not_accepted():
    raw,clean=inputs()
    with pytest.raises(ValueError,match='genuine'):
        StdShapeAugment()(raw[:,:,:64,:64],clean)


def test_erasure_guard_falls_back_clean():
    raw,clean=inputs()
    good,_=guards(torch.zeros_like(clean),clean,torch.zeros(4))
    assert not good.any()
    good,_=guards(clean,clean,torch.ones(4)*.1)
    assert not good.any()


def test_width_jitter_direction_and_no_64px_binary_operation():
    raw,_=inputs(2);x=raw.float()/255
    out=width_jitter(x,torch.tensor([1.,-1.]))
    assert out[0].sum()>x[0].sum() and out[1].sum()<x[1].sum()


def test_bounded_gate_and_zero_paths_after_bias_drift():
    m=tiny(std_soft_bias=False,glyph_scale_bound=.5,glyph_scale_init=.2,glyph_vec_cond=True)
    assert float(m.effective_glyph_scale().detach())==pytest.approx(.2)
    assert all(c.bias is None for c in m.glyph_embedder if isinstance(c,torch.nn.Conv2d))
    with torch.no_grad():
        m.glyph_embedder[-1].weight.normal_()
        for mod in m.glyph_vec_proj.modules():
            if hasattr(mod,'bias') and mod.bias is not None:mod.bias.fill_(3.)
        m.glyph_scale.fill_(100)
    assert float(m.effective_glyph_scale().detach())<=.5
    x=torch.randn(2,256,64)
    xx,tok,vec=embed_glyph(EmbedGlyphCfg.from_model(m),x,torch.zeros(2,1,64,64),None,
        torch.zeros(2,dtype=torch.long),None,None,None,None)
    assert torch.equal(xx,x) and torch.count_nonzero(tok)==0 and torch.count_nonzero(vec)==0


def test_legacy_defaults_state_and_forward_unchanged():
    torch.manual_seed(4);a=tiny()
    torch.manual_seed(4);b=tiny(std_soft_bias=True,glyph_scale_bound=0.)
    assert a.state_dict().keys()==b.state_dict().keys()
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    b.load_state_dict(a.state_dict(),strict=True)


def test_20l_factory_same_width_and_explicit_route():
    m=DiT_2Cond_models['DiT-2Cond-S384-20/2'](num_calligraphers=10,num_characters=12,
        use_char_cond=False,use_glyph_cond=True,std_soft_cond=True,glyph_in_channels=1,
        content_cond='std',cond_inject_at='3,6,9,12,15',std_soft_bias=False,
        glyph_scale_bound=.5,glyph_scale_init=.2)
    assert len(m.blocks)==20 and m.x_embedder.hidden_size==384
    assert m.cond_router.layers==[3,6,9,12,15]
    assert m.glyph_embedder[-1].weight.count_nonzero()==0


def test_stage_configs_same_horizon_and_no_hidden_training_losses():
    from src.train.cli import parse_args
    for mode in ('strong','mild'):
        c=configuration(mode)
        assert c['model']=='DiT-2Cond-S384-20/2' and c['max_steps']==200000 and c['stop_at_step']==5000
        assert c['pretrained'] is None and not c['warm_start_from'] and not c['resume_full']
        assert c['pair_fraction']==c['rank_gradient_ratio']==c['w_critic_writer']==c['w_critic_away']==0
    assert configuration('strong',50000)['max_steps']==configuration('strong',5000)['max_steps']


def test_training_eval_build_strict_checkpoint_roundtrip(tmp_path,monkeypatch):
    from src.eval.model_io import load_model_from_ckpt
    from src.model.build import build_dit_from_args
    # Keep the production20L factory assertion above; use a tiny registry entry to test the shared I/O path.
    monkeypatch.setitem(DiT_2Cond_models,'test-v90',lambda **kw:DiT_2Cond(depth=2,hidden_size=64,num_heads=4,patch_size=2,**kw))
    c=configuration();c.update(model='test-v90',cond_inject_at='0,1',triple_table_prefix='',freeze_triple_tables=False,
        num_calligraphers=10,num_characters=12)
    args=SimpleNamespace(**c);m=build_dit_from_args(args,device='cpu')
    path=tmp_path/'checkpoint.pt';torch.save({'model':m.state_dict(),'args':args},path)
    restored,_=load_model_from_ckpt(str(path),device='cpu',use_ema=False)
    assert restored.std_soft_bias is False and restored.glyph_scale_bound==.5
    assert restored.glyph_vec_cond
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in m.state_dict().items())


def test_resume_uses_actual_train_steps_key_and_preserves_horizon(tmp_path):
    from tools.analysis.validate_v90_launch import validate
    c=configuration(stop=50000);cache=tmp_path/'cache.pt';cache.touch();c['std_shape_cache']=str(cache)
    args=SimpleNamespace(**c)
    old=dict(c,stop_at_step=5000)
    checkpoint=dict(train_steps=5000,args=old,opt={'state':{}},scheduler={'last_epoch':5000})
    validate(args,checkpoint)
    with pytest.raises(ValueError,match='phase step'):
        validate(args,dict(checkpoint,train_steps=3000))
    wrong=dict(old,global_batch_size=64)
    with pytest.raises(ValueError,match='identity'):
        validate(args,dict(checkpoint,args=wrong))


def test_dataset_wrapper_preserves_gt_and_rejects_wrong_font_cache(tmp_path):
    from src.utils.std_shape_data import StdShapeDataset
    import hashlib
    root=tmp_path/'store';root.mkdir();(root/'index.csv').write_text('manifest',encoding='utf-8')
    class Base:
        samples=[dict(script_id='0',character='X')]
        _std_soft=SimpleNamespace(root=str(root))
        target=torch.randn(4,32,32)
        def __len__(self):return 1
        def __getitem__(self,i):return dict(latent=self.target,std_soft=torch.ones(1,64,64))
    base=Base();cache=tmp_path/'cache.pt';raw=torch.ones(1,1,256,256,dtype=torch.uint8)
    data=dict(version=2,source_kind='precomputed_std_shape',keys=[(0,'X')],variants=raw,
        source_index_sha256=hashlib.sha256((root/'index.csv').read_bytes()).hexdigest())
    torch.save(data,cache);wrapped=StdShapeDataset(base,cache)
    assert torch.equal(wrapped[0]['latent'],base.target)
    assert wrapped[0]['std_shape_index']==0
    # Do not overwrite an mmap'd file (Windows correctly keeps the source read-only).
    data['source_index_sha256']='invalid';bad_cache=tmp_path/'bad.pt';torch.save(data,bad_cache)
    with pytest.raises(ValueError,match='index disagree'):StdShapeDataset(base,bad_cache)
