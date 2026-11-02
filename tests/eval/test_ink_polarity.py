# -*- coding: utf-8 -*-
"""Ink-polarity and topology-metric direction tests.

Why this file exists: "dark means ink" was written backwards in three separate scripts in
one session (a topology_split unit check, p05_invert, and ts_window_audit). Each time the
failure was SILENT -- the metric either went to zero or inverted, and the resulting
numbers looked like experimental findings. Asserting the reference mask is non-empty is
not enough: an inverted but non-empty mask passes that check.

So each property here is asserted as a DIRECTION with a known-correct answer, using
synthetic images where the right response is unambiguous:

  1. all-white input  -> ink probability ~0
  2. all-black input  -> ink probability ~1
  3. soft mask and hard threshold agree in direction
  4. manufactured break / merge / erase each move the metrics the right way

These catch the class of bug by construction rather than by inspection.
"""
import sys

import numpy as np
import pytest

sys.path.insert(0, ".")
from src.eval.metrics import ink_retention, topology_split  # noqa: E402


def ink_prob(img01, k=12.0, mid=0.5):
    """Canonical ink probability for an image in [0,1]; ink is DARK.

    Accepts HWC or BCHW. Note the axis handling: for HWC the channels are LAST, so the
    channel mean is over axis=-1 -- averaging over axis=1 there would average over image
    ROWS and silently produce a wrongly-shaped, meaningless map. This mirrors the
    convention every caller must use, so the tests pin the definition rather than each
    script re-deriving it (and getting the sign or the axis wrong).
    """
    import torch
    t = torch.as_tensor(img01, dtype=torch.float32)
    if t.ndim == 3:                                   # HWC -> BCHW
        t = t.permute(2, 0, 1)[None]
    return torch.sigmoid((mid - t.mean(dim=1, keepdim=True)) * k)


def make(char="一", h=64, w=64):
    """A simple dark stroke on white, in [0,1] HWC."""
    a = np.ones((h, w, 3), dtype=np.float32)
    a[h // 2 - 3:h // 2 + 3, 8:w - 8] = 0.0          # horizontal bar
    if char == "二":
        a[h // 5:h // 5 + 3, 8:w - 8] = 0.0
    return a


# ── 1 & 2: the polarity contract ──────────────────────────────────────────────
def test_all_white_gives_no_ink():
    white = np.ones((64, 64, 3), dtype=np.float32)
    p = ink_prob(white)
    assert float(p.mean()) < 0.01, f"white must read as ~0 ink, got {float(p.mean()):.4f}"
    assert float(p.max()) < 0.01, "no pixel of a white image may read as ink"


def test_all_black_gives_all_ink():
    black = np.zeros((64, 64, 3), dtype=np.float32)
    p = ink_prob(black)
    assert float(p.mean()) > 0.99, f"black must read as ~1 ink, got {float(p.mean()):.4f}"
    assert float(p.min()) > 0.99, "every pixel of a black image must read as ink"


def test_inverted_convention_would_fail_these():
    """Guard against 'fixing' ink_prob by flipping the sign: show the flip is detectable."""
    white = np.ones((64, 64, 3), dtype=np.float32)
    wrong = 1.0 - ink_prob(white)
    assert float(wrong.mean()) > 0.99, "the inverted convention is exactly what these tests catch"


# ── 3: soft and hard agree in direction ───────────────────────────────────────
def test_soft_and_hard_masks_agree():
    img = make()
    p = ink_prob(img)
    soft_high = (p > 0.5)
    hard_ink = (img.mean(axis=2) < 0.5)
    # Wherever the hard mask says ink, the soft mask must too (and vice versa).
    agree = float((soft_high[0, 0].numpy() == hard_ink).mean())
    assert agree > 0.98, f"soft/hard masks must agree in direction, agree={agree:.4f}"
    assert hard_ink.sum() > 0, "sanity: the synthetic stroke must produce ink pixels"


# ── 4: each metric moves the right way under a manufactured defect ────────────
def test_break_increases_component_excess():
    gt = make()
    broken = gt.copy()
    broken[32 - 3:32 + 3, 28:34] = 1.0               # cut the stroke in half
    s = topology_split(broken[None], gt[None])
    assert s["component_excess"] == 1.0, "a split stroke must register as EXCESS"
    assert s["frag"] > 1.5, f"a split stroke must raise frag, got {s['frag']:.3f}"


def test_merge_increases_component_deficit():
    gt = make("二")                                   # two strokes
    merged = np.ones_like(gt)
    merged[12:52, 8:56] = 0.0                        # one blob covering both
    s = topology_split(merged[None], gt[None])
    assert s["component_deficit"] == 1.0, "fusing strokes must register as DEFICIT"
    assert s["frag"] < 0.8, f"fusing strokes must lower frag, got {s['frag']:.3f}"


def test_erasure_must_be_caught_by_ink_retention_not_frag():
    """The lesson from P0.5: erasing strokes LOWERS frag and looks like a repair.

    frag alone cannot distinguish 'repaired' from 'erased', so any acceptance test that
    relies on frag must also report ink retention.
    """
    gt = make()
    erased = np.ones_like(gt)                        # stroke removed entirely
    s = topology_split(erased[None], gt[None])
    assert s["frag"] < 1.0, "erasure must LOWER frag -- this is why frag alone is unsafe"
    ret = ink_retention(erased[None], gt[None])
    assert ret == 0.0, "erasure must show up as ~0 ink retention"
    # and the deficit is what names it: too FEW components, not a repair
    assert s["component_deficit"] == 1.0


def test_identical_pair_is_neutral():
    gt = make()
    s = topology_split(gt[None], gt[None])
    assert s["frag"] == pytest.approx(1.0)
    assert s["component_excess"] == 0.0 and s["component_deficit"] == 0.0
    assert s["parity_frac"] == 1.0


def test_two_error_modes_can_cancel_in_a_single_average():
    """The reason topology_split exists: demonstrate the cancellation concretely."""
    gt = make()
    broken = gt.copy()
    broken[32 - 3:32 + 3, 28:34] = 1.0
    broken_pair = topology_split(broken[None], gt[None])["frag"]
    erased_pair = topology_split(np.ones_like(gt)[None], gt[None])["frag"]
    assert broken_pair > 1.0 and erased_pair < 1.0, "the two modes pull frag opposite ways"
    # averaged over the pair, the two opposite errors nearly cancel
    combined = 0.5 * (broken_pair + erased_pair)
    assert abs(combined - 1.0) < abs(broken_pair - 1.0), (
        "averaging over- and under-fragmentation must hide both, which is why the "
        "aggregate frag is not a safe headline number")
