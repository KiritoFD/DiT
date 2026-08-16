import unittest
import csv
import tempfile
from pathlib import Path

import numpy as np
import torch

from latent_dataset import MCCDLatentDataset
from models import DiT_3Cond
from samplers import DistributedFactorBalancedSampler


class TinyDataset:
    samples = [
        {"character_id": "0", "calligrapher_id": "0"},
        {"character_id": "1", "calligrapher_id": "1"},
        {"character_id": "1", "calligrapher_id": "1"},
        {"character_id": "1", "calligrapher_id": "1"},
    ]

    def __len__(self):
        return len(self.samples)


class Factorized3CondTest(unittest.TestCase):
    def make_model(self, drop_all=0.0, drop_one=0.0):
        return DiT_3Cond(
            input_size=4, patch_size=2, hidden_size=48, depth=2, num_heads=4,
            num_calligraphers=3, num_scripts=2, num_characters=5,
            condition_fusion="factorized_add", callig_embed_dim=12,
            script_embed_dim=8, char_embed_dim=16,
            cond_drop_all_prob=drop_all, cond_drop_one_prob=drop_one)

    def test_forward_and_cfg_shapes(self):
        model = self.make_model().eval()
        for batch in (1, 3):
            x = torch.randn(batch, 4, 4, 4)
            t = torch.randint(0, 1000, (batch,))
            yc = torch.zeros(batch, dtype=torch.long)
            ys = torch.zeros(batch, dtype=torch.long)
            yh = torch.ones(batch, dtype=torch.long)
            self.assertEqual(model(x, t, yc, ys, yh).shape, (batch, 8, 4, 4))
            self.assertEqual(model.forward_with_cfg(x, t, yc, ys, yh).shape,
                             (batch, 8, 4, 4))

    def test_drop_all_matches_explicit_null_ids(self):
        model = self.make_model(drop_all=1.0).train()
        x = torch.randn(2, 4, 4, 4)
        t = torch.randint(0, 1000, (2,))
        labels = torch.tensor([0, 1])
        dropped = model(x, t, labels, labels, labels)
        model.eval()
        null_output = model(
            x, t,
            torch.full_like(labels, model.y_callig_embedder.num_classes),
            torch.full_like(labels, model.y_script_embedder.num_classes),
            torch.full_like(labels, model.y_char_embedder.num_classes))
        torch.testing.assert_close(dropped, null_output)

    def test_balanced_sampler_is_deterministic_and_upweights_tail(self):
        sampler = DistributedFactorBalancedSampler(TinyDataset(), seed=7)
        self.assertGreater(sampler.weights[0], sampler.weights[1])
        first = list(iter(sampler))
        self.assertEqual(first, list(iter(sampler)))
        sampler.set_epoch(1)
        self.assertNotEqual(first, list(iter(sampler)))

    def test_non_preload_latent_lookup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            latent = np.arange(4 * 2 * 2, dtype=np.float32).reshape(1, 4, 2, 2)
            np.savez(root / "shard_00000.npz", img_ids=np.array([42]), latents=latent)
            csv_path = root / "data.csv"
            with open(csv_path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "image_path", "calligrapher_id", "script_id", "character_id"])
                writer.writeheader()
                writer.writerow({"image_path": "final_images/42.png",
                                 "calligrapher_id": 0, "script_id": 0,
                                 "character_id": 0})
            dataset = MCCDLatentDataset(
                str(csv_path), str(root), img_root=None, preload=False,
                load_image=False)
            torch.testing.assert_close(dataset._get_latent(42), torch.from_numpy(latent[0]))


if __name__ == "__main__":
    unittest.main()
