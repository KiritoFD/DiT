import torch
from torch import nn

from src.unet.io import Config,build
from src.unet.blocks import ResBlock,Attention
from src.unet.variants.regularization import schedule_scale,drop_residual,optimizer_parameters


def test_depth_schedule_and_eval_identity():
    assert [schedule_scale(s) for s in (0,2000,3500,5000,9000)]==[0.,0.,.5,1.,1.]
    x=torch.ones(1024,2,2,2)
    assert drop_residual(x,.08,training=False) is x
    assert torch.equal(drop_residual(x,.08,scale=torch.tensor(0.)),x)
    out=drop_residual(x,.5)
    assert set(out.unique().tolist())=={0.,2.}
    assert torch.equal(out[:,0,0,0],out[:,1,1,1])
    assert abs(float(out.mean())-1)<.1


def test_extreme_model_and_weight_decay_partition():
    with torch.device('meta'):
        model=build(Config(channels=(96,192,256),num_res_blocks=8,num_writers=20,drop_path_max=.08))
    assert sum(isinstance(m,ResBlock) for m in model.modules())==50
    assert sum(isinstance(m,Attention) for m in model.modules())==5
    assert model.encoder[0].blocks[0].drop_path_rate==0.
    assert model.decoder[0].blocks[-1].drop_path_rate==.08
    assert not model.writer.weight.requires_grad
    groups=optimizer_parameters(model,Config(grouped_weight_decay=True,weight_decay=.05))
    decayed={id(p) for p in groups[0]['params']};plain={id(p) for p in groups[1]['params']}
    assert not decayed&plain
    assert decayed|plain=={id(p) for p in model.parameters() if p.requires_grad}
    assert id(model.input.weight) in decayed
    assert id(model.input.bias) in plain and id(model.script.weight) in plain
    assert id(model.encoder[0].blocks[0].norm1.weight) in plain
    assert id(model.writer.weight) not in decayed|plain
