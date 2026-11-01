"""Small regression checks; no checkpoints, downloads or GPU needed."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
import torch
from torch import nn
from .common import GuidedVelocity


class Contracts(unittest.TestCase):
    def test_saturated_ink_matching(self):
        import numpy as np
        from .score import match_ink
        ink=np.zeros((32,32),dtype=np.float32);ink[8:24,8:24]=1
        for target in (.15,.35):
            self.assertAlmostEqual(float(match_ink(ink,target).mean()),target,places=6)
    def test_labels(self):
        from tools.u100_pretrain.eval_offline import rewrite
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);src=p/'raw.csv'
            with src.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['calligrapher_id']);w.writeheader();w.writerow({'calligrapher_id':601})
            rewrite(src,p/'first.csv');rewrite(p/'first.csv',p/'second.csv')
            self.assertEqual((p/'first.csv').read_bytes(),(p/'second.csv').read_bytes())
    def test_nested_and_done_metrics(self):
        from tools.u100_pretrain.eval_offline import extract_metrics
        values=dict(ssim=.5,lpips=.3,ink_iou=.2,n=100)
        self.assertEqual(extract_metrics({'eval200fix':values},'',1,100),values)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'eval_samples_ctrl/step0000001/eval200fix';p.mkdir(parents=True)
            (p/'metrics.json').write_text(json.dumps(values))
            self.assertEqual(extract_metrics({},d,1,100),values)
            with self.assertRaises(ValueError):extract_metrics({},d,1,99)
    def test_native_time_and_cfg(self):
        from src.loss.flow_matching import FlowMatching
        class Oracle(nn.Module):
            num_writers=10
            def __init__(self):super().__init__();self.times=[]
            def forward(self,x,t,y_callig,**kw):
                self.times.extend(t.tolist());return torch.ones_like(x)*(y_callig!=10)[:,None,None,None]
        base=Oracle();net=GuidedVelocity(base,1.5)
        process=FlowMatching(num_steps=2,sampler='heun')
        z=process.ddim_sample_loop(net,(1,4,2,2),x_T=torch.zeros(1,4,2,2),device='cpu',model_kwargs={'y_callig':torch.zeros(1,dtype=torch.long)})
        self.assertTrue(torch.allclose(z,torch.full_like(z,-1.5)))
        self.assertEqual(max(base.times),1000.);self.assertEqual(min(base.times),0.)
        self.assertEqual(net.branch_sample_evals,8)


if __name__=='__main__':unittest.main()
