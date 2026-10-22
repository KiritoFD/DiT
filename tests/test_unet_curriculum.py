"""Focused curriculum contracts; tiny CPU models, no full training."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from src.unet.io import Config,build
from src.unet.curriculum.transfer import initialize
from src.unet.curriculum.runtime import Adaptation,groups
from src.unet.curriculum.data import ReplaySampler
from src.train.ema import update_ema
from tools.generic_pretrain.contracts import NearIndex,normalized


class Curriculum(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def config(self,**kwargs):
        return Config(**dict(channels=(64,64,64),num_res_blocks=1,num_writers=2,writer_mode='none',
            decouple_writer_condition=True,**kwargs))

    def inputs(self):
        return (torch.randn(2,4,32,32),torch.tensor([10.,100.]),torch.tensor([0,1]),
                dict(g=torch.rand(2,1,64,64),y_script=torch.tensor([0,1])))

    def test_generic_bypasses_writer_even_invalid_ids(self):
        model=build(self.config()).eval();x,t,w,k=self.inputs()
        handle=model.writer.register_forward_hook(lambda *a:self.fail('Writer lookup executed'))
        self.assertTrue(torch.equal(model(x,t,w,**k),model(x,t,None,**k)))
        model(x,t,w,**k).sum().backward()
        self.assertIsNone(model.writer.weight.grad);self.assertIsNone(model.writer_proj.weight.grad)
        handle.remove()

    def test_generic_retains_std_and_script_signal(self):
        torch.manual_seed(90)
        model=build(self.config()).eval();x,t,w,k=self.inputs()
        # A zero output head would make all conditional effects trivially zero.
        for name,p in model.named_parameters():
            if name.endswith(('conv2.weight','output.2.weight')):torch.nn.init.normal_(p,std=.02)
        k['g'].requires_grad_(True)
        original=model(x,t,w,**k)
        script_changed=model(x,t,w,**{**k,'y_script':1-k['y_script']})
        std_changed=model(x,t,w,**{**k,'g':torch.zeros_like(k['g'])})
        self.assertFalse(torch.equal(original,script_changed))
        self.assertFalse(torch.equal(original,std_changed))
        original.square().mean().backward()
        self.assertGreater(float(k['g'].grad.abs().sum()),0.)
        self.assertGreater(float(model.script.weight.grad.abs().sum()),0.)
        self.assertGreater(float(model.script_proj.weight.grad.abs().sum()),0.)
        self.assertIsNone(model.writer.weight.grad)

    def test_strict_transfer_zero_equivalence_and_null_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);mapping=path/'map.json';mapping.write_text('{}')
            old=self.config(font_remap_json=str(mapping),char_remap_json=str(mapping))
            source=build(old).eval()
            # Nontrivial pretrained residuals/output; otherwise zero head hides a broken transfer.
            for name,p in source.named_parameters():
                if name.endswith(('conv2.weight','output.2.weight')):torch.nn.init.normal_(p,std=.02)
            checkpoint=path/'pre.pt'
            torch.save(dict(architecture=old.architecture,config=asdict(old),train_steps=900,
                            ema=source.state_dict()),checkpoint)
            config=Config(**{**asdict(old),'writer_mode':'film','writer_film_rank':4,
                'writer_film_layers':['middle','decoder'],'initialize_from':str(checkpoint)})
            target=build(config).eval();record=initialize(target,config)
            x,t,w,k=self.inputs();base=source(x,t,w,**k)
            self.assertTrue(record['new_clock'])
            self.assertTrue(torch.equal(base,target(x,t,w,**k)))
            # Before adding synthetic nonzero biases, the zero-init branch must already learn.
            for name,p in target.named_parameters():p.requires_grad_('.writer_film.' in name)
            opt=torch.optim.SGD([p for p in target.parameters() if p.requires_grad],lr=.1)
            target(x,t,w,**k).square().mean().backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for n,p in target.named_parameters() if '.writer_film.2.' in n))
            opt.step();target.zero_grad()
            changed=target(x,t,w,**k)
            swapped=target(x,t,1-w,**k)
            self.assertFalse(torch.equal(changed,swapped))
            self.assertTrue(torch.equal(base,target(x,t,torch.full_like(w,2),**k)))
            # Trained FiLM biases cannot leak into UNKNOWN/null Writer replay.
            for name,p in target.named_parameters():
                if '.writer_film.2.bias' in name:torch.nn.init.constant_(p,.4)
            null=target(x,t,torch.full_like(w,2),**k)
            self.assertTrue(torch.equal(base,null))
            self.assertGreater(float((target(x,t,w,**k)-null).abs().max().detach()),0.)
            target.zero_grad();target(x,t,w,**k).square().mean().backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for n,p in target.named_parameters() if '.writer_film.' in n))

    def test_unfreeze_refreshes_ema_and_keeps_optimizer_groups(self):
        c=Config(**{**asdict(self.config()),'writer_mode':'film','writer_film_rank':4,
                    'writer_film_layers':['middle','decoder'],'adapter_only_steps':2,'std_shape_aug':'off'})
        model=build(c);ema=copy.deepcopy(model).requires_grad_(False);opt=torch.optim.AdamW(groups(model,c))
        adaptation=Adaptation(model,ema,c,None)
        self.assertTrue(all('.writer_film.' in n for n,p in model.named_parameters() if p.requires_grad))
        update_ema(ema,model,.5);self.assertTrue(hasattr(ema,'_ema_pairs'))
        adaptation.set_phase(2);self.assertFalse(hasattr(ema,'_ema_pairs'))
        self.assertTrue(model.output[-1].weight.requires_grad)
        self.assertFalse(model.input.weight.requires_grad)
        self.assertFalse(model.writer.weight.requires_grad)
        original=ema.output[-1].weight.clone()
        with torch.no_grad():model.output[-1].weight.add_(1.)
        update_ema(ema,model,.5)
        self.assertTrue(torch.allclose(ema.output[-1].weight,original+.5))
        self.assertEqual([g['lr'] for g in opt.param_groups],[c.lr,c.lr*.2])

    def test_replay_is_probability_not_dataset_size(self):
        rows=[{'generic':'1'}]*900+[{'generic':'0'}]*100
        sampler=ReplaySampler(rows,100,100,.1,90)
        sample=list(sampler);ratio=sum(i<900 for i in sample)/len(sample)
        self.assertAlmostEqual(ratio,.1,delta=.015)
        self.assertEqual(sample,list(sampler));sampler.set_epoch(1)
        self.assertNotEqual(sample,list(sampler))

    def test_protected_near_index_and_unknown_labels(self):
        index=NearIndex();thumb=np.arange(4096,dtype=np.uint8).reshape(64,64)
        index.add({'phash':'0'},thumb)
        self.assertTrue(index.matches({'phash':str(255)},thumb))
        row=normalized(dict(character='永',script_id='0',calligrapher_id='123'),True)
        self.assertEqual(row['calligrapher_id'],'-1');self.assertEqual(row['generic'],'1')

if __name__=='__main__':unittest.main()
