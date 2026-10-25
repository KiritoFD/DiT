"""Small contract checks, not a broad training/metric regression suite."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from src.unet.reference.encoder import StyleHead, supcon
from src.unet.reference.data import ReferenceBank, QuartetSampler
from src.unet.model import LatentUNet


class ReferenceStyleTests(unittest.TestCase):
    def test_full_batch_supcon_and_gradient(self):
        head=StyleHead()
        embedding=head(torch.randn(4,768))
        loss,stats=supcon(embedding,torch.tensor([0,0,1,1]),torch.tensor([0,1,0,1]),torch.zeros(4))
        loss.backward()
        self.assertEqual(int(stats['valid_anchors']),4)
        self.assertEqual(int(stats['hard_negatives']),4)
        self.assertGreater(float(head.net[0].weight.grad.norm()),0)

    def test_reference_excludes_all_target_char_images(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'bank.npz'
            np.savez(path,ids=np.arange(5),features=np.ones((5,768)),
                     writers=[0,0,0,1,1],scripts=np.zeros(5),chars=[0,0,1,0,1])
            bank=ReferenceBank(path)
            refs,mask=bank.sample(np.array([0,1]),np.random.default_rng(90),2)
            self.assertTrue((bank.chars[refs][mask]==1).all())
            samples=[dict(img_id=i) for i in range(5)]
            batch=list(QuartetSampler(samples,bank,4,1,90))
            p=bank.positions(batch)
            self.assertEqual(len(set(bank.writers[p])),2)
            self.assertEqual(len(set(bank.chars[p])),2)

    def test_writer_id_does_not_condition_generator(self):
        model=LatentUNet(channels=(64,64,64),num_res_blocks=1,writer_mode='none',
                         decouple_writer_condition=True,reference_dim=128).eval()
        # Nonzero output head makes the equality check nontrivial.
        torch.nn.init.normal_(model.output[-1].weight,std=.01)
        x=torch.randn(1,4,32,32);t=torch.tensor([.5]);refs=torch.randn(1,2,768)
        kwargs=dict(g=torch.rand(1,1,64,64),y_script=torch.tensor([0]),reference_features=refs)
        with torch.no_grad():
            a=model(x,t,torch.tensor([0]),**kwargs)
            b=model(x,t,torch.tensor([9]),**kwargs)
        self.assertTrue(torch.equal(a,b))
        with self.assertRaises(ValueError):model(x,t,torch.tensor([0]),y_script=torch.tensor([0]))


if __name__=='__main__':unittest.main()
