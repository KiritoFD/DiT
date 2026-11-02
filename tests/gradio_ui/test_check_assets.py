"""The asset checker must fail loudly and specifically, not optimistically.

There is currently no U-Net-format checkpoint on this machine, so the checker cannot be
validated against a real serveable model here -- and a checker that has only ever printed
"missing" is untested in the direction that matters. These tests therefore build the
conditions synthetically: a fake checkpoint with a known config, and asset paths that are
deliberately absent, so each branch is exercised.

The important property under test is that the tool never reports ok=True for something it
did not actually verify. A false green here would send a user to a broken UI.
"""
import json
from pathlib import Path

import torch

from tools.gradio_ui.check_assets import check_one


def _write_ckpt(tmp_path, *, architecture='latent_cnn_unet_v1', with_config=True):
    p = tmp_path / 'fake.pt'
    # A minimal but structurally valid config: everything the loader needs to rebuild the
    # module. Asset paths point at files that do not exist, which is what the tests assert
    # on -- the point is that a present checkpoint with missing assets is reported, not
    # masked by a load failure.
    config = dict(
        architecture='latent_cnn_unet_v1', channels=(64, 128, 192), num_res_blocks=1,
        num_writers=4, num_scripts=3, style_matrix_repr=2,
        style_matrix_injection=1,
        std_soft_root='does/not/exist/std_bbox64', vae_path='does/not/exist/vae',
        # Field names follow Config exactly. `latent_channels` and `std_soft_cond` are NOT
        # Config fields (they live on the eval args), so they are absent here on purpose.
        callig_remap_json='does/not/exist/callig.json',
        font_remap_json='does/not/exist/font.json',
        data_csv='does/not/exist/train.csv', eval_csv='does/not/exist/eval.csv',
    )
    if with_config:
        # Build the real module so the state dict is complete; an empty one would be
        # rejected by load()'s strict=True and test the wrong failure.
        from src.unet.io import Config, build
        weights = build(Config(**config)).state_dict()
    else:
        weights = {}
    torch.save(dict(architecture=architecture, ema=weights, delta=weights,
                    train_steps=123, config=config if with_config else {}, args={}),
               str(p))
    return p


def test_missing_checkpoint_is_reported_not_crashed(tmp_path):
    r = check_one(str(tmp_path / 'nope.pt'), 'cpu', sample=False)
    assert r['ok'] is False
    assert any('missing' in p for p in r['problems'])


def test_legacy_checkpoint_without_architecture_is_refused(tmp_path):
    """A DiT-era file has no `architecture`, and load() must refuse rather than guess."""
    p = _write_ckpt(tmp_path, with_config=False)
    torch.save(dict(ema={'x': torch.zeros(1)}, delta={}, train_steps=1), str(p))
    r = check_one(str(p), 'cpu', sample=False)
    assert r['ok'] is False
    assert any('LOAD FAILED' in pb for pb in r['problems'])


def test_dit_architecture_is_refused(tmp_path):
    """The legacy gradio app serves a DiT; this path must reject it explicitly."""
    p = _write_ckpt(tmp_path, architecture='DiT-2Cond-S/2')
    r = check_one(str(p), 'cpu', sample=False)
    assert r['ok'] is False
    assert any('DiT' in pb or 'ARCHITECTURE' in pb or 'Refuse' in pb
               for pb in r['problems'])


def test_missing_assets_are_listed_by_name(tmp_path):
    """Every absent dependency must be named, so the fix is a path not a guess."""
    p = _write_ckpt(tmp_path)
    r = check_one(str(p), 'cpu', sample=False)
    assert r['ok'] is False
    joined = ' '.join(r['problems'])
    assert 'vae_path' in joined
    assert 'std_soft_root' in joined
    assert 'callig_remap_json' in joined


def test_std_soft_root_is_always_required(tmp_path):
    """Conditioning is unconditional in `eval_args`, so the glyph cache is never optional.

    This replaces an earlier test that assumed a `std_soft_cond=False` path existed in the
    checkpoint config. It does not: `std_soft_cond` is not a Config field at all, so
    reading it off `cfg` returns None and would silently skip this check.
    """
    p = _write_ckpt(tmp_path)
    r = check_one(str(p), 'cpu', sample=False)
    assert 'std_soft_root' in r['checks']
    assert r['checks']['std_soft_root']['ok'] is False


def test_report_never_claims_ok_without_passing_every_check(tmp_path):
    p = _write_ckpt(tmp_path)
    r = check_one(str(p), 'cpu', sample=False)
    # ok must be exactly the negation of "has problems"
    assert r['ok'] == (not r['problems'])
    if r['problems']:
        assert r['ok'] is False


def test_report_is_json_serialisable(tmp_path):
    """The report is written to disk and consumed by the UI, so no odd objects."""
    p = _write_ckpt(tmp_path)
    r = check_one(str(p), 'cpu', sample=False)
    json.dumps(r)
