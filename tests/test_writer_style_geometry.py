import numpy as np
import torch
from src.model.writer_style_latent import WriterStyleLatent,augment_latent,warp_geometry_sdf
from src.model.writer_style_geometry import WriterStyleGeometry
from tools.analysis.prepare_writer_geometry_sdf import signed_distance


def test_spatial_geometry_and_embedding_shapes():
    torch.set_num_threads(2);torch.manual_seed(89);model=WriterStyleGeometry();o=model(torch.randn(2,4,32,32))
    assert o['geometry_shapes']==((32,32),(16,16),(8,8))
    assert o['geometry_sdf'].shape==(2,1,32,32)
    assert o['embedding'].shape==(2,256) and o['slot_logits'].shape==(2,23)
    assert torch.allclose(o['embedding'].norm(dim=1),torch.ones(2),atol=1e-5)
    ratio=sum(p.numel() for p in model.parameters())/sum(p.numel() for p in WriterStyleLatent().parameters())
    assert .9<ratio<1.1


def test_d2_d3_initialization_and_augmentation_are_paired():
    torch.manual_seed(89);a=WriterStyleGeometry()
    torch.manual_seed(89);b=WriterStyleGeometry()
    for x,y in zip(a.parameters(),b.parameters()):assert torch.equal(x,y)
    z=torch.randn(4,4,32,32)
    torch.manual_seed(12);plain=augment_latent(z)
    torch.manual_seed(12);aug,grid,scale=augment_latent(z,return_transform=True)
    assert torch.equal(plain,aug)
    assert grid.shape==(4,32,32,2) and scale.shape==(4,)


def test_sdf_sign_and_identity_warp():
    gray=np.full((128,128),255,np.uint8);gray[40:88,40:88]=0
    sdf=torch.tensor(signed_distance(gray))[None].float()
    assert sdf[0,0,16,16]>0 and sdf[0,0,0,0]<0
    matrix=torch.tensor([[[1.,0,0],[0,1.,0]]]);grid=torch.nn.functional.affine_grid(matrix,sdf.shape,align_corners=False)
    assert torch.allclose(warp_geometry_sdf(sdf,grid,torch.ones(1)),sdf.clamp(-1,1),atol=1e-6)


def test_uniform_scale_corrects_sdf_values_before_clipping():
    sdf=torch.full((1,1,32,32),.4);matrix=torch.tensor([[[.8,0,0],[0,.8,0]]]);grid=torch.nn.functional.affine_grid(matrix,sdf.shape,align_corners=False)
    assert torch.allclose(warp_geometry_sdf(sdf,grid,torch.tensor([.8])),torch.full_like(sdf,.5),atol=1e-6)


def test_geometry_aux_reaches_spatial_branch_not_writer_projection():
    model=WriterStyleGeometry();o=model(torch.randn(2,4,32,32));o['geometry_sdf'].square().mean().backward()
    assert model.geometry[2][0].body[0].weight.grad.norm()>0
    assert model.stem[0].weight.grad.norm()>0
    assert model.project[1].weight.grad is None
    assert model.brush[0][0].body[0].weight.grad is None
