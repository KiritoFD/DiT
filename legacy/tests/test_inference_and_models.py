"""
Unit tests for DiT_2Cond 2-Axis CFG, inference module, and CalligraphySampler.
"""

import sys
import os
import unittest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models import DiT_2Cond_S_4, DiT_2Cond_B_4, DiT_2Cond_models
from src.inference import CalligraphySampler, SCRIPT_NAMES_TO_ID


class TestModelsAndInference(unittest.TestCase):
    def test_dit_2cond_s4_construction_and_shapes(self):
        """Test DiT-2Cond-S/4 with kl-f4 shape (in_channels=3, input_size=16)"""
        model = DiT_2Cond_S_4(
            input_size=16,
            in_channels=3,
            num_calligraphers=1011,
            num_characters=35130,
            condition_fusion="factorized_add",
            callig_embed_dim=128,
            char_embed_dim=768,
        )
        model.eval()

        B = 2
        x = torch.randn(B, 3, 16, 16)
        t = torch.tensor([100, 200], dtype=torch.long)
        yc = torch.tensor([130, 956], dtype=torch.long)
        yg = torch.tensor([2175, 4500], dtype=torch.long)

        # Standard forward
        out = model(x, t, yc, yg)
        self.assertEqual(out.shape, (B, 6, 16, 16))

        # Standard CFG forward
        out_cfg = model.forward_with_cfg(x, t, yc, yg, cfg_scale=4.0)
        self.assertEqual(out_cfg.shape, (B, 6, 16, 16))

        # 2-Axis CFG forward
        out_2axis = model.forward_with_2axis_cfg(
            x, t, yc, yg, cfg_callig=2.0, cfg_glyph=4.0, w_inter=0.0
        )
        self.assertEqual(out_2axis.shape, (B, 6, 16, 16))

    def test_dit_2cond_b4_construction_and_shapes(self):
        """Test DiT-2Cond-B/4 with kl-f4 shape (in_channels=3, input_size=16)"""
        model = DiT_2Cond_B_4(
            input_size=16,
            in_channels=3,
            num_calligraphers=1011,
            num_characters=35130,
            condition_fusion="factorized_add",
            callig_embed_dim=128,
            char_embed_dim=768,
        )
        model.eval()

        B = 3
        x = torch.randn(B, 3, 16, 16)
        t = torch.tensor([50, 150, 250], dtype=torch.long)
        yc = torch.tensor([10, 20, 30], dtype=torch.long)
        yg = torch.tensor([100, 200, 300], dtype=torch.long)

        out_2axis = model.forward_with_2axis_cfg(
            x, t, yc, yg, cfg_callig=1.5, cfg_glyph=3.5, w_inter=0.25
        )
        self.assertEqual(out_2axis.shape, (B, 6, 16, 16))

    def test_id_resolution(self):
        """Test script & name resolution mappings in inference"""
        self.assertEqual(SCRIPT_NAMES_TO_ID["楷"], 0)
        self.assertEqual(SCRIPT_NAMES_TO_ID["草"], 2)
        self.assertEqual(SCRIPT_NAMES_TO_ID["隶书"], 4)


if __name__ == "__main__":
    unittest.main()
