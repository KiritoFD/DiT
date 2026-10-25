import unittest
import torch
import numpy as np
from src.unet.io import Config,build
from src.unet.brush.attributes import measure,normalize


class BrushContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_width_and_invalid(self):
        narrow=np.zeros((128,128),np.float32);narrow[8:120,60:66]=1
        wide=np.zeros_like(narrow);wide[8:120,56:70]=1
        a,ok=measure(narrow);b,ok2=measure(wide)
        self.assertTrue(ok and ok2);self.assertGreater(b[1],a[1])
        self.assertFalse(measure(np.zeros_like(narrow))[1])

    def test_train_only_normalization(self):
        x=np.ones((303,5),np.float32);x[-3:]=1000
        script=np.r_[np.repeat(np.arange(3),100),np.arange(3)]
        train=np.arange(303)<300;valid=np.ones(303,bool)
        z,center,scale=normalize(x,script,train,valid)
        self.assertTrue(np.array_equal(center,np.ones((3,5))))

    def test_brush_zero_null_gradient_and_writer_cfg(self):
        torch.manual_seed(90)
        c=Config(channels=(64,64,64),num_res_blocks=1,writer_mode='film',decouple_writer_condition=True,
            writer_film_rank=4,writer_film_layers=('middle','decoder.'),brush_dim=5)
        model=build(c).eval()
        for name,p in model.named_parameters():
            if name.endswith(('conv2.weight','output.2.weight')):torch.nn.init.normal_(p,std=.02)
        x=torch.randn(2,4,32,32);t=torch.ones(2)*.5;w=torch.zeros(2,dtype=torch.long)
        kw=dict(g=torch.rand(2,1,64,64),y_script=torch.zeros(2,dtype=torch.long))
        attrs=torch.randn(2,5)
        zero=model(x,t,w,brush_attrs=attrs,**kw)
        self.assertTrue(torch.equal(zero,model(x,t,w,brush_attrs=attrs+1,**kw)))
        zero.square().mean().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters() if '.brush_film.' in n))
        with torch.no_grad():
            for n,p in model.named_parameters():
                if '.brush_film.weight' in n:torch.nn.init.normal_(p,std=.01)
                if '.brush_film.bias' in n:p.fill_(.2)
        invalid=torch.zeros(2,dtype=torch.bool)
        self.assertTrue(torch.equal(zero,model(x,t,w,brush_attrs=attrs,brush_valid=invalid,**kw)))
        self.assertFalse(torch.equal(model(x,t,w,brush_attrs=attrs,**kw),model(x,t,w,brush_attrs=attrs+1,**kw)))
        # Zero Writer branches: CFG must not accidentally change Brush into a different default.
        with torch.no_grad():model.brush_defaults[0].fill_(.8)
        self.assertTrue(torch.allclose(model.forward_with_cfg(x,t,w,cfg_scale=2,**kw),
            model.forward_with_cfg(x,t,w,cfg_scale=1,**kw),atol=1e-6))


if __name__=='__main__':unittest.main()
