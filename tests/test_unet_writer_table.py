import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from src.unet.io import Config,build,checkpoint_config
from src.unet.table_contract import install_writer_table
from src.train.ema import update_ema


def test_new_models_freeze_writer_and_legacy_checkpoint_identity_stays_explicit():
    with torch.device('meta'):
        model=build(Config(channels=(64,128,192),num_res_blocks=1))
    assert not model.writer.weight.requires_grad
    legacy=checkpoint_config(dict(architecture='latent_cnn_unet_v1',config={'channels':[64,128,192]},args={}))
    assert legacy.freeze_writer_table is False


def test_resume_replacement_clears_momentum_and_freezes_raw_and_ema(tmp_path):
    model=nn.Module();model.writer=nn.Embedding(3,32);model.projection=nn.Linear(32,1)
    ema=copy.deepcopy(model).requires_grad_(False)
    opt=torch.optim.AdamW(model.parameters(),lr=.1,weight_decay=.1)
    model.projection(model.writer(torch.tensor([0,1]))).sum().backward();opt.step();opt.zero_grad(set_to_none=True)
    assert model.writer.weight in opt.state
    null=model.writer.weight[-1].detach().clone()
    table=np.full((2,32),.1234567,dtype=np.float32);np.save(tmp_path/'callig_table.npy',table)
    (tmp_path/'callig_remap.json').write_text(json.dumps({'4':0,'8':1}))
    c=SimpleNamespace(writer_table_file=str(tmp_path/'callig_table.npy'),freeze_writer_table=True,
                      callig_remap_json=str(tmp_path/'callig_remap.json'))
    install_writer_table(model,ema,opt,c)
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        model.projection(model.writer(torch.tensor([0,1]))).sum().backward()
        opt.step();update_ema(ema,model,.998001)
    assert model.writer.weight.grad is None and model.writer.weight not in opt.state
    assert torch.equal(model.writer.weight[:-1],torch.from_numpy(table))
    assert torch.equal(model.writer.weight,ema.writer.weight)
    assert torch.equal(model.writer.weight[-1],null)


def test_external_table_must_not_be_trainable():
    with pytest.raises(ValueError,match='frozen'):
        Config(freeze_writer_table=False,writer_table_file='x.npy').validate(require_budget=False)


def test_dino_cls_handles_dict_like_transformers_output():
    from tools.writer_table.pretrain import cls_feature
    class Output(dict):
        pass
    out=Output();out.last_hidden_state=torch.randn(2,3,8)
    out['last_hidden_state']=out.last_hidden_state
    assert torch.equal(cls_feature(out),out.last_hidden_state[:,0])
    expected=torch.randn(2,8)
    assert cls_feature({'x_norm_clstoken':expected}) is expected
