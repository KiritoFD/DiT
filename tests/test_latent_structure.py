import unittest

import torch

from latent_structure import LatentStructureLoss, LatentStructureProbe
from latent_condition_probe import LatentConditionProbe


class LatentStructureTests(unittest.TestCase):
    def test_condition_probe_compact_head_shapes(self):
        probe = LatentConditionProbe(
            num_characters=17, num_calligraphers=7, num_scripts=3, width=16)
        char, callig, script = probe(torch.randn(2, 4, 32, 32))
        self.assertEqual(char.shape, (2, 17))
        self.assertEqual(callig.shape, (2, 7))
        self.assertEqual(script.shape, (2, 3))

    def test_canny_loss_backpropagates_without_decoder(self):
        pred = torch.randn(4, 4, 32, 32, requires_grad=True)
        target = torch.randn_like(pred)
        canny = torch.zeros(4, 1, 256, 256)
        canny[:, :, 64:192, 126:130] = 1
        losses = LatentStructureLoss(max_timestep=500)(
            pred, target, torch.tensor([10, 400, 600, 999]), canny=canny)
        self.assertGreater(losses["canny"].item(), 0)
        losses["canny"].backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertEqual(pred.grad[2:].abs().sum().item(), 0)

    def test_frozen_skeleton_probe_keeps_input_gradient(self):
        probe = LatentStructureProbe(width=16, depth=1)
        loss_fn = LatentStructureLoss(probe=probe, max_timestep=999).train()
        self.assertFalse(probe.training)
        self.assertFalse(any(parameter.requires_grad for parameter in probe.parameters()))
        pred = torch.randn(2, 4, 32, 32, requires_grad=True)
        skeleton = torch.zeros(2, 1, 256, 256)
        skeleton[:, :, 32:224, 127:129] = 1
        loss = loss_fn(pred, torch.zeros_like(pred), torch.tensor([1, 2]),
                       skeleton=skeleton)["skeleton"]
        loss.backward()
        self.assertGreater(pred.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
