"""Regression tests for the raw 64x64 standard-glyph conditioning path."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from src.eval.inference import (make_eval_cache, sample_latents,
                                validate_std_soft_condition)
from src.model.dit import DiT_2Cond
from src.utils.latent_dataset import MCCDLatentDataset
from src.utils.std_soft_glyph import StdSoftGlyphStore


def _make_store(root: Path, pairs=((0, "永"),)) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for script_id, char in pairs:
        rel = Path(str(script_id)) / f"U+{ord(char):05X}.png"
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("L", (64, 64), 0)
        ImageDraw.Draw(image).line((12, 12, 52, 52), fill=255, width=5)
        image.save(path)
        rows.append({"script_id": script_id, "character": char, "path": rel.as_posix()})
    with (root / "index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("script_id", "character", "path"))
        writer.writeheader()
        writer.writerows(rows)
    return root


def test_store_returns_blurred_single_channel_tensor(tmp_path):
    # Arrange
    store = StdSoftGlyphStore(
        str(_make_store(tmp_path / "store")), is_train=False,
        eval_sigma=0.75, preload=True, strict=True)

    # Act
    glyph = store.get(0, "永")

    # Assert
    assert glyph.shape == (1, 64, 64)
    assert glyph.dtype == torch.float32
    assert 0.0 <= float(glyph.min()) <= float(glyph.max()) <= 1.0
    assert torch.any((glyph > 0) & (glyph < 1)), "Gaussian blur should create soft edges"
    with pytest.raises(KeyError, match="std-soft glyph missing"):
        store.get(3, "永")


def test_eval_cache_uses_fixed_std_soft_condition(tmp_path):
    # Arrange
    store_root = _make_store(tmp_path / "store")
    gt_path = tmp_path / "gt.png"
    Image.new("RGB", (32, 32), "white").save(gt_path)
    csv_path = tmp_path / "eval.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "image_path", "img_id", "calligrapher_id", "script_id",
            "character", "character_id"))
        writer.writeheader()
        writer.writerow({
            "image_path": str(gt_path), "img_id": 1, "calligrapher_id": 0,
            "script_id": 0, "character": "永", "character_id": 0})

    # Act
    cache = make_eval_cache(
        str(csv_path), None, None, image_size=32, n=1, vae_downscale=8,
        latent_channels=4, scaling_factor=0.18215,
        std_soft_root=str(store_root), std_soft_sigma=0.65)

    # Assert
    assert cache["g_kind"] == "std_soft"
    assert cache["skels_latent"].shape == (1, 1, 64, 64)
    assert cache["missing_skel"] == 0


def test_latent_dataset_exposes_std_soft_without_legacy_g(tmp_path):
    # Arrange
    store_root = _make_store(tmp_path / "store")
    shard_root = tmp_path / "latents"
    shard_root.mkdir()
    np.savez(
        shard_root / "shard_00000.npz",
        img_ids=np.asarray([1], dtype=np.int64),
        latents=np.zeros((1, 4, 32, 32), dtype=np.float32))
    csv_path = tmp_path / "train.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "image_path", "img_id", "calligrapher_id", "script_id",
            "character", "character_id"))
        writer.writeheader()
        writer.writerow({
            "image_path": "1.png", "img_id": 1, "calligrapher_id": 0,
            "script_id": 0, "character": "永", "character_id": 0})
    dataset = MCCDLatentDataset(
        str(csv_path), str(shard_root), img_root="", load_image=False,
        use_glyph_cond=False, std_soft_root=str(store_root),
        is_train=False, std_soft_eval_sigma=0.65)

    # Act
    sample = dataset[0]

    # Assert
    assert sample["std_soft"].shape == (1, 64, 64)
    assert sample["g"].numel() == 0


def _tiny_model(**kwargs):
    return DiT_2Cond(
        input_size=32, patch_size=2, in_channels=4,
        hidden_size=48, depth=2, num_heads=4, mlp_ratio=2.0,
        num_calligraphers=3, num_characters=5, learn_sigma=False,
        condition_fusion="legacy", **kwargs)


def test_std_soft_encoder_matches_image_token_grid_and_is_zero_initialized():
    # Arrange
    model = _tiny_model(
        use_glyph_cond=True, std_soft_cond=True,
        std_soft_zero_init=True, glyph_in_channels=1)
    x = torch.randn(2, 4, 32, 32)
    t = torch.full((2,), 500.0)
    y_callig = torch.tensor([0, 1])
    y_char = torch.tensor([1, 2])
    g = torch.rand(2, 1, 64, 64)

    # Act
    g_tokens = model.glyph_embedder(g)
    out = model(x, t, y_callig, y_char, g=g)

    # Assert
    assert g_tokens.shape == (2, 48, 16, 16)
    assert torch.count_nonzero(model.glyph_embedder[-1].weight) == 0
    assert out.shape == x.shape
    with pytest.raises(ValueError, match="glyph token grid mismatch"):
        model(x, t, y_callig, y_char, g=torch.rand(2, 1, 32, 32))


def test_legacy_glyph_encoder_contract_is_unchanged():
    # Arrange / Act
    model = _tiny_model(
        use_glyph_cond=True, std_soft_cond=False, glyph_in_channels=4)

    # Assert
    assert isinstance(model.glyph_embedder, torch.nn.Conv2d)
    assert model.glyph_embedder.in_channels == 4


@pytest.mark.parametrize("bad", [
    None, torch.zeros(2, 4, 32, 32), torch.zeros(2, 1, 64, 64),
    torch.full((2, 1, 64, 64), float("nan")),
    torch.full((2, 1, 64, 64), -0.1), torch.full((2, 1, 64, 64), 1.1),
    torch.ones(2, 1, 64, 64, dtype=torch.uint8),
])
def test_std_soft_rejects_missing_or_invalid_condition(bad):
    with pytest.raises(ValueError, match="std-soft"):
        validate_std_soft_condition(bad, 2)


class _RecordingStdSoftModel:
    std_soft_cond = True  # Also exercise models without use_glyph_cond.
    in_channels = 4

    def __init__(self):
        self.received = []

    def forward_with_cfg(self, x, t, y_callig, y_char, cfg_scale, g=None):
        self.received.append(g.clone())
        return x


class _OneStepSampler:
    def ddim_sample_loop(self, model, shape, noise, **kwargs):
        return model(noise, torch.zeros(shape[0]), **kwargs["model_kwargs"])


def test_sampling_routes_std_soft_to_g_in_every_batch():
    model = _RecordingStdSoftModel()
    g = torch.rand(3, 1, 64, 64)
    noise = torch.rand(3, 4, 32, 32)
    result = sample_latents(model, _OneStepSampler(), noise, [(0, 0)] * 3,
                            cfg_scale=1.0, batch=2, device=torch.device("cpu"), skel=g)
    assert torch.equal(result, noise)
    assert torch.equal(torch.cat(model.received), g)


def test_sampling_fails_before_sampler_when_std_soft_is_missing():
    with pytest.raises(ValueError, match="nonempty g"):
        sample_latents(_RecordingStdSoftModel(), _OneStepSampler(),
                       torch.zeros(2, 4, 32, 32), [(0, 0)] * 2,
                       cfg_scale=1.0, batch=2, device=torch.device("cpu"))


def test_cache_rejects_legacy_cache_for_std_soft(monkeypatch):
    from argparse import Namespace
    from src.eval import in_mem_eval
    monkeypatch.setattr(in_mem_eval, "_CACHES", {})
    monkeypatch.setattr(in_mem_eval, "make_eval_cache", lambda *a, **kw: {
        "n": 1, "skels_latent": None})
    args = Namespace(std_soft_cond=True, std_soft_root="store", vae_scaling_factor=.18215)
    with pytest.raises(ValueError, match="legacy condition cache"):
        in_mem_eval._get_cache("unused.csv", 1, None, "", args)


def test_standalone_eval_overrides_stored_gamma():
    """覆盖 stored gamma ✓ 并设置正确的快链路评测参数 ✓。"""
    from argparse import Namespace
    from tools.eval_std_soft_checkpoint import prepare_eval_args
    args = Namespace(std_soft_cond=True, sde_gamma=.5)
    prepare_eval_args(args, gamma=0.0, steps=50, cfg=1.0, batch=64)
    assert args.sde_gamma == 0.0
    assert args.in_mem_eval_lpips is True
    assert args.in_mem_eval_batch == 64
    assert args.in_mem_eval_vae_batch == 64
    with pytest.raises(ValueError, match="gamma/cfg must be nonnegative"):
        prepare_eval_args(args, gamma=-0.5, steps=50, cfg=1.0, batch=64)

