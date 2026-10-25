"""Comprehensive unit tests for PixelWriterV2, Haar DWT, Metric Losses, and Quartets."""
import pytest
import torch
import torch.nn.functional as F
import numpy as np

from src.model.writer_style_pixel import PixelWriterV2, haar_dwt_2d, haar_dwt_1d, erase_ink_keep_background
from src.loss.metric_learning import compute_metric_loss, same_script_ce_loss, euclid_contrastive_loss
from src.utils.writer_quartets import WriterQuartets, build_same_script_mask


def test_haar_dwt_2d_properties():
    """Verify 2D Haar DWT sub-band shapes and mathematical reconstruction property."""
    x = torch.randn(2, 1, 64, 64)
    ll, lh, hl, hh = haar_dwt_2d(x)
    assert ll.shape == (2, 1, 32, 32)
    assert lh.shape == (2, 1, 32, 32)
    assert hl.shape == (2, 1, 32, 32)
    assert hh.shape == (2, 1, 32, 32)

    # Inverse Haar DWT check:
    # x00 = (ll - lh - hl + hh) / 2
    # x01 = (ll - lh + hl - hh) / 2
    # x10 = (ll + lh - hl - hh) / 2
    # x11 = (ll + lh + hl + hh) / 2
    x00_rec = (ll - lh - hl + hh) * 0.5
    x01_rec = (ll - lh + hl - hh) * 0.5
    x10_rec = (ll + lh - hl - hh) * 0.5
    x11_rec = (ll + lh + hl + hh) * 0.5

    x_rec = torch.zeros_like(x)
    x_rec[:, :, 0::2, 0::2] = x00_rec
    x_rec[:, :, 0::2, 1::2] = x01_rec
    x_rec[:, :, 1::2, 0::2] = x10_rec
    x_rec[:, :, 1::2, 1::2] = x11_rec

    assert torch.allclose(x, x_rec, atol=1e-5), "2D Haar DWT must be perfectly invertible!"


def test_pixel_writer_v2_forward_and_haar_modes():
    """Verify PixelWriterV2 forward pass across none, 1d, and 2d Haar modes."""
    B = 4
    x = torch.rand(B, 1, 128, 128)

    for mode in ('none', '1d', '2d'):
        model = PixelWriterV2(n_slots=23, haar_mode=mode)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out['embedding'].shape == (B, 256)
        # Verify L2 normalization
        norms = torch.norm(out['embedding'], dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
        assert out['slot_logits'].shape == (B, 23)

    # Test head reset
    model.reset_head(n_slots=45)
    assert model.n_slots == 45
    assert model(x)['slot_logits'].shape == (B, 45)


def test_same_script_ce_masking():
    """Verify that same-script logit masking strictly isolates cross-script gradients."""
    targets = torch.tensor([0, 1, 2], dtype=torch.long)
    slot_scripts = [0, 0, 1, 1]  # slots 0,1 are script 0; slots 2,3 are script 1
    mask = build_same_script_mask(targets, slot_scripts, n_slots=4)

    # For sample 0 (target 0, script 0): slots 0 and 1 must be True, 2 and 3 must be False
    assert mask[0].tolist() == [True, True, False, False]
    # For sample 2 (target 2, script 1): slots 0 and 1 must be False, 2 and 3 must be True
    assert mask[2].tolist() == [False, False, True, True]

    logits = torch.randn(3, 4, requires_grad=True)
    loss = same_script_ce_loss(logits, targets, script_mask=mask)
    loss.backward()

    # Gradients for masked-out slots (e.g., slots 2,3 for sample 0) must be virtually 0
    assert torch.allclose(logits.grad[0, 2:], torch.zeros(2), atol=1e-6)
    assert torch.allclose(logits.grad[2, :2], torch.zeros(2), atol=1e-6)


def test_euclidean_contrastive_loss():
    """Verify Euclidean contrastive loss behavior and gradient flow."""
    # Create 4-sample quartet: 0 and 1 share writer A, 2 and 3 share writer B
    # Sample 0 & 2 have char X; Sample 1 & 3 have char Y
    chars = ['X', 'Y', 'X', 'Y']
    # Perfect embedding: pos distance = 0, neg distance = 2.0
    e = torch.tensor([
        [1.0, 0.0],
        [1.0, 0.0],
        [-1.0, 0.0],
        [-1.0, 0.0]
    ], requires_grad=True)

    loss, stats = euclid_contrastive_loss(e, chars, n_quartet_samples=4, margin=1.0)
    # pos_dist = 0, neg_dist = 2 > margin 1.0 -> loss should be 0!
    assert loss.item() == pytest.approx(0.0, abs=1e-4)

    # Suboptimal embedding: pos distance large
    e_bad = torch.tensor([
        [1.0, 0.0],
        [-1.0, 0.0],
        [1.0, 0.0],
        [-1.0, 0.0]
    ], requires_grad=True)
    loss_bad, stats_bad = euclid_contrastive_loss(e_bad, chars, n_quartet_samples=4, margin=1.0)
    assert loss_bad.item() > 0.5
    loss_bad.backward()
    assert e_bad.grad is not None and torch.isfinite(e_bad.grad).all()


def test_background_probe_erasing():
    """Verify that erase_ink_keep_background erases dark stroke pixels."""
    canvas = torch.ones(2, 1, 32, 32) * 0.9  # Light paper background
    canvas[:, :, 10:20, 10:20] = 0.1         # Dark ink patch
    erased = erase_ink_keep_background(canvas, threshold=0.5)

    # Ink patch should be raised close to 0.9 paper background
    assert erased[:, :, 10:20, 10:20].mean().item() > 0.8
