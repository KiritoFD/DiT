import inspect
import numpy as np
import pytest
import torch
from src.loss.crosschar_style import CrossCharacterFeatureSampler,cross_character_nce


def test_all_positives_are_different_char_and_negatives_are_same_script_other_writer():
    w=np.array([0,0,0,1,1,1,2,2,2])
    s=np.zeros(9,dtype=int)
    c=np.array(['x','y','z']*3)
    work=np.array(['a','b','c']*3)
    sampler=CrossCharacterFeatureSampler(w,s,c,work)
    p,n,stats=sampler.sample(w,s,c,work)
    assert np.all(w[p]==w) and np.all(c[p]!=c)
    assert np.all(w[n]!=w[:,None]) and np.all(s[n]==s[:,None])
    assert np.all(c[n]==c[:,None])
    assert stats['hard_negative_fraction']==1
    assert stats['positive_cross_source_fraction']==1


def test_no_self_positive_or_wrong_script_fallback():
    sampler=CrossCharacterFeatureSampler([0,1],[0,0],['x','x'],['a','b'])
    with pytest.raises(ValueError):
        sampler.sample([0],[0],['x'],['a'])


def test_reference_features_stop_gradient_but_query_remains_differentiable():
    q=torch.randn(3,16,requires_grad=True)
    bank=torch.randn(9,16,requires_grad=True)
    loss=cross_character_nce(q,bank,torch.tensor([0,1,2]),torch.tensor([[3,4],[5,6],[7,8]]))
    loss.backward()
    assert q.grad.abs().sum()>0 and torch.isfinite(q.grad).all()
    assert bank.grad is None


def test_inference_has_no_reference_input_and_gt_replacement_cannot_change_it():
    from src.model.dit import DiT_2Cond
    forbidden={'gt','gt_image','gt_latent','reference_style','style_features','reference_features'}
    assert not forbidden & set(inspect.signature(DiT_2Cond.forward).parameters)
    kw=dict(input_size=8,patch_size=2,hidden_size=32,depth=2,num_heads=4,
        num_calligraphers=2,num_scripts=1,callig_embed_dim=8,script_embed_dim=4,
        use_script_cond=True,use_char_cond=False,content_cond='std',std_soft_cond=True,
        use_glyph_cond=True,glyph_in_channels=1,condition_fusion='factorized_cat',learn_sigma=False)
    model=DiT_2Cond(**kw).eval()
    with torch.no_grad():
        model.final_layer.linear.weight.normal_(std=.02)
    x=torch.randn(2,4,8,8);t=torch.tensor([100.,500.])
    conditions=dict(y_callig=torch.tensor([0,1]),y_script=torch.zeros(2,dtype=torch.long),
                    y_char=torch.zeros(2,dtype=torch.long),g=torch.rand(2,1,16,16))
    expected=model(x,t,**conditions).detach().clone()
    # The only mutation is in an EXTERNAL loss-side bank, not model inputs/state.
    loss_bank=torch.randn(20,256)
    loss_bank.normal_(mean=100,std=20)
    del loss_bank
    assert torch.equal(expected,model(x,t,**conditions))
    assert not any('prototype' in k or 'teacher' in k or 'reference' in k for k in model.state_dict())
    with pytest.raises(TypeError):
        model(x,t,**conditions,reference_style=torch.randn(2,256))


def test_loss_side_teacher_uses_whole_batch_and_restores_generator_mode(tmp_path):
    import hashlib
    import logging
    from types import SimpleNamespace
    from src.model.style_encoder import build_style_encoder
    from src.train.crosschar_style import CrossCharacterPostTraining
    ckpt=tmp_path/'teacher.pt';bank_path=tmp_path/'bank.pt'
    teacher=build_style_encoder(4,1,'local_stats')
    torch.save({'model':teacher.state_dict(),'slots':['A·s','B·s','C·s','std·s'],
                'scripts':['s'],'architecture':'local_stats'},ckpt)
    matched=tmp_path/'matched_teacher.json';matched.write_text('{"test_fixture":true}')
    split=tmp_path/'split_indices.npz'
    np.savez(split,train=np.arange(9),val_char=np.array([9]),test_char=np.array([10]),test_source=np.array([11]))
    batch_size=12
    target={i:(i%3,0,['x','y','z'][i%3],'a') for i in range(batch_size)}
    torch.save({'encoder_sha256':hashlib.sha256(ckpt.read_bytes()).hexdigest(),
                'partition':'encoder_train_only','features':torch.randn(9,256),
                'teacher_accepted':True,'matched_teacher_sha256':hashlib.sha256(matched.read_bytes()).hexdigest(),
                'split_sha256':hashlib.sha256(split.read_bytes()).hexdigest(),'reference_indices':np.arange(9),
                'encoder_script_ids':torch.tensor([0]),'target_metadata':target,
                'writers':np.repeat(np.arange(3),3),'scripts':np.zeros(9,dtype=int),
                'chars':np.array(['x','y','z']*3),'sources':np.array(['b']*9)},bank_path)
    args=SimpleNamespace(diffusion_type='flow',crosschar_bank=str(bank_path),
        crosschar_encoder_ckpt=str(ckpt),crosschar_gradient_ratio=.2,strong_style_max_weight=10.)
    auxiliary=CrossCharacterPostTraining(args,'cpu',logging.getLogger('crosschar-test'))
    class Generator(torch.nn.Module):
        def __init__(self):
            super().__init__();self.conv=torch.nn.Conv2d(4,4,1);self.observed_batch=None
        def forward(self,x,t,y_callig,y_script,y_char,g):
            self.observed_batch=len(x)
            return self.conv(x)
    model=Generator().train()
    batch={'y_callig_raw':torch.arange(batch_size)%3,'y_script':torch.zeros(batch_size,dtype=torch.long),
           'img_id':torch.arange(batch_size),'std_soft':torch.rand(batch_size,1,64,64)}
    value=auxiliary(model,torch.randn(batch_size,4,32,32),torch.rand(batch_size),
                    {'y_char':torch.zeros(batch_size,dtype=torch.long)},batch,
                    torch.tensor(.1),0,fm_norm=torch.tensor(.3))
    value.backward()
    assert model.observed_batch==batch_size and model.training
    assert model.conv.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in auxiliary.teacher.parameters())
    payload=torch.load(bank_path,weights_only=False)
    payload['reference_indices']=np.array([10,*range(1,9)])
    torch.save(payload,bank_path)
    with pytest.raises(ValueError,match='Held-out GT'):
        CrossCharacterPostTraining(args,'cpu',logging.getLogger('crosschar-test'))
