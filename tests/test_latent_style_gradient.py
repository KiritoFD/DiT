import numpy as np
import torch
import pytest
import hashlib
import zipfile
from tools.analysis.latent_style_gradient import bounded_step,rms,reference_ids,contrastive_per_image
from tools.analysis.latent_style_gradient import prototypes
from tools.analysis.latent_style_gradient import validate_pretrained_archive


def test_step_has_per_image_relative_rms_bound():
    torch.manual_seed(89);z=torch.randn(5,4,32,32);g=torch.randn_like(z)
    y=bounded_step(z,g,.01)
    assert torch.allclose(rms(y-z)/rms(z),torch.full((5,),.01),atol=1e-6)
    assert ((y-z)*g).flatten(1).sum(1).lt(0).all()


def test_zero_gradient_does_not_change_latent():
    z=torch.randn(2,4,32,32)
    assert torch.equal(z,bounded_step(z,torch.zeros_like(z),.03))


def test_references_are_train_only_and_content_matched():
    data={'splits':{'train':np.array([0,1,2])},'writer':np.array(['A','A','B','B']),
          'script':np.array(['s']*4),'char':np.array(['x','y','x','x'])}
    pos,neg=reference_ids(data,'A','s','x')
    assert pos.tolist()==[1] and neg.tolist()==[2]


def test_contrastive_gradient_reaches_input_not_frozen_parameters():
    model=torch.nn.Linear(4,4,bias=False).requires_grad_(False)
    z=torch.randn(2,4,requires_grad=True);e=torch.nn.functional.normalize(model(z),dim=1)
    pos=torch.nn.functional.normalize(torch.randn(2,4),dim=1)
    neg=torch.nn.functional.normalize(torch.randn(2,3,4),dim=2);mask=torch.tensor([[1,0,0],[1,1,0]],dtype=torch.bool)
    loss=contrastive_per_image(e,pos,neg,mask).mean();g=torch.autograd.grad(loss,z)[0]
    assert torch.isfinite(g).all() and g.norm()>0
    assert all(p.grad is None for p in model.parameters())


def test_prototype_excludes_query_character_and_all_heldout_rows():
    data={'splits':{'train':np.array([0,1,2,3])},'labels':np.array([0,0,1,1,0]),
          'char':np.array(['x','y','x','y','y']),'source':np.array(['a','a','b','b','holdout']),
          'slots':[('A','s'),('B','s')]}
    x=np.array([[100,100],[1,0],[200,200],[0,1],[1e6,1e6]],np.float32)
    metadata=[{'script':'s','character':'x'}]*2
    p,m=prototypes(x,data,metadata)
    assert torch.equal(p[0],torch.eye(2))
    assert torch.equal(p[0],p[1]) and m.all()


def test_pretrained_cache_rejects_wrong_hash_and_truncated_zip(tmp_path):
    path=tmp_path/'weights.pth'
    with zipfile.ZipFile(path,'w') as z:z.writestr('data.pkl',b'test')
    good=hashlib.sha256(path.read_bytes()).hexdigest()
    assert validate_pretrained_archive(path,good)==good
    with pytest.raises(ValueError,match='SHA256'):validate_pretrained_archive(path,'0'*64)
    path.write_bytes(path.read_bytes()[:-8]);bad=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(zipfile.BadZipFile):validate_pretrained_archive(path,bad)
